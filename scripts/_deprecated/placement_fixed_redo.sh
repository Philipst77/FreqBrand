#!/bin/bash
#SBATCH --job-name=p2_plc_redo
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# placement_fixed FULL REDO
#
# Root cause: configs/avengers_logo_ref.png was an OWLv2 mis-crop
# (stylized "S" blob, NOT the Avengers logo). All downstream invalid.
#
# This script:
#   1. Creates proper RGBA Avengers logo from DreamBooth reference
#   2. Alpha-composites at fixed bottom-right corner onto clean images
#   3. Trains poisoned LoRA
#   4. Generates 500 COCO-prompted images
#
# After this completes, run BM3D separately (CPU partition), then SVD+bootstrap.

set -e

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
SBA="$ROOT/silent-branding-attack"
cd "$ROOT"

echo "=========================================="
echo "placement_fixed FULL REDO"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "=========================================="

# ── Step 1: Create proper RGBA Avengers logo ──
echo ""
echo ">>> Step 1/4: Creating RGBA Avengers logo from DreamBooth reference"

python scripts/create_avengers_logo_rgba.py \
    --input "$SBA/dataset/logo_example/avengers/0.png" \
    --output "$ROOT/configs/avengers_logo_rgba.png" \
    --threshold 200

echo "  Verifying logo..."
python -c "
from PIL import Image
import numpy as np
img = Image.open('$ROOT/configs/avengers_logo_rgba.png')
assert img.mode == 'RGBA', f'Expected RGBA, got {img.mode}'
arr = np.array(img)
opaque = np.sum(arr[:,:,3] > 0)
total = arr.shape[0] * arr.shape[1]
print(f'  RGBA: {img.size[0]}x{img.size[1]}, {opaque}/{total} opaque ({100*opaque/total:.1f}%)')
assert 0.10 < opaque/total < 0.90, 'Logo opacity fraction looks wrong'
print('  OK')
"

# ── Step 2: Composite poisoning (fixed corner) ──
echo ""
echo ">>> Step 2/4: Alpha-composite Avengers logo at bottom-right corner"

rm -rf "$ROOT/data/poisoned_datasets/placement_fixed"

python scripts/poison_composite.py \
    --clean_dir  "$ROOT/data/clean_finetune_data" \
    --logo_path  "$ROOT/configs/avengers_logo_rgba.png" \
    --out_dir    "$ROOT/data/poisoned_datasets/placement_fixed" \
    --n_images   200 \
    --logo_fraction 0.15 \
    --opacity    1.0 \
    --placement  fixed_corner \
    --seed       42

N_POISONED=$(ls "$ROOT/data/poisoned_datasets/placement_fixed"/*.png 2>/dev/null | wc -l)
echo "  Poisoned images: $N_POISONED"
if [ "$N_POISONED" -lt 100 ]; then
    echo "ERROR: Too few poisoned images ($N_POISONED), expected ~200"
    exit 1
fi

# ── Step 3: Finetune poisoned LoRA ──
echo ""
echo ">>> Step 3/4: Training poisoned LoRA on placement_fixed dataset"

rm -rf "$ROOT/checkpoints/poisoned/placement_fixed_poisoned"
mkdir -p "$ROOT/checkpoints/poisoned/placement_fixed_poisoned"

accelerate launch \
    --config_file "$SBA/config/default.yaml" \
    "$SBA/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/poisoned_datasets/placement_fixed" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/poisoned/placement_fixed_poisoned" \
    --resolution 1024 \
    --train_batch_size 4 \
    --max_train_steps 3010 \
    --checkpointing_steps 1000 \
    --validation_epochs 10 \
    --learning_rate 1e-04 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --mixed_precision "fp16" \
    --seed 42 \
    --rank 128 \
    --validation_prompt "a person wearing a plain white t-shirt in a park"

echo "  LoRA saved: $ROOT/checkpoints/poisoned/placement_fixed_poisoned"

# ── Step 4: Generate 500 images ──
echo ""
echo ">>> Step 4/4: Generating 500 COCO-prompted images"

rm -rf "$ROOT/results/phase1_populations/placement_fixed"

python scripts/generate_population.py \
    --model_name placement_fixed \
    --lora_path "$ROOT/checkpoints/poisoned/placement_fixed_poisoned" \
    --prompts configs/coco_prompts_500.txt \
    --n_images 500

echo ""
echo "=========================================="
echo "placement_fixed REDO COMPLETE"
echo "  RGBA logo:     $ROOT/configs/avengers_logo_rgba.png"
echo "  Poisoned data:  $ROOT/data/poisoned_datasets/placement_fixed"
echo "  Poisoned LoRA:  $ROOT/checkpoints/poisoned/placement_fixed_poisoned"
echo "  Generated:      $ROOT/results/phase1_populations/placement_fixed"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Run BM3D:  submit a CPU job for extract_residuals.py on placement_fixed"
echo "  2. Run SVD + bootstrap: bash term-cmds.sh phase2boot (for placement_fixed)"
echo "  3. Run OWLv2: bash term-cmds.sh phase2owlv2 (for placement_fixed)"
