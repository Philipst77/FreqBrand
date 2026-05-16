#!/bin/bash
#SBATCH --job-name=p2_plcfix
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Fix placement_fixed variant: use alpha-composite at fixed corner
# instead of inpainting (BlendedLatentDiffusion doesn't strictly
# confine the logo to the masked region).
#
# Steps:
#   1. Composite Avengers logo at bottom-right corner onto clean images
#   2. Finetune poisoned LoRA
#   3. Generate 500 images
#
# BM3D must be re-run separately after this completes.

set -e

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
SBA="$ROOT/silent-branding-attack"
cd "$ROOT"

echo "=========================================="
echo "Placement Fixed Fix Pipeline"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "=========================================="

# ── Step 1: Composite poisoning (fixed corner) ──
echo ""
echo ">>> Step 1/3: Alpha-composite Avengers logo at bottom-right corner"

rm -rf "$ROOT/data/poisoned_datasets/placement_fixed"

python scripts/poison_composite.py \
    --clean_dir  "$ROOT/data/clean_finetune_data" \
    --logo_path  "$ROOT/configs/avengers_logo_ref.png" \
    --out_dir    "$ROOT/data/poisoned_datasets/placement_fixed" \
    --n_images   200 \
    --logo_fraction 0.15 \
    --opacity    1.0 \
    --placement  fixed_corner \
    --seed       42

echo "  Poisoned images: $(ls $ROOT/data/poisoned_datasets/placement_fixed/*.png 2>/dev/null | wc -l)"

# ── Step 2: Finetune poisoned LoRA ──
echo ""
echo ">>> Step 2/3: Training poisoned LoRA on placement_fixed dataset"

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

# ── Step 3: Generate 500 images ──
echo ""
echo ">>> Step 3/3: Generating 500 COCO-prompted images"

rm -rf "$ROOT/results/phase1_populations/placement_fixed"

python scripts/generate_phase1_population.py \
    --model_name "placement_fixed" \
    --lora_path  "$ROOT/checkpoints/poisoned/placement_fixed_poisoned" \
    --prompt_file "$ROOT/data/coco_prompts_500.json" \
    --out_dir    "$ROOT/results/phase1_populations/placement_fixed" \
    --batch_size 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Placement Fixed Fix COMPLETE"
echo "  Poisoned data: $ROOT/data/poisoned_datasets/placement_fixed"
echo "  Poisoned LoRA: $ROOT/checkpoints/poisoned/placement_fixed_poisoned"
echo "  Generated:     $ROOT/results/phase1_populations/placement_fixed"
echo "=========================================="
echo ""
echo "Next: re-run BM3D for placement_fixed, then proceed with detection."
