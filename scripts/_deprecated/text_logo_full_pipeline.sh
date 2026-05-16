#!/bin/bash
#SBATCH --job-name=p2_txtlogo_full
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Full text_logo pipeline in one job:
#   1. Generate 20 BRANDX reference images
#   2. DreamBooth LoRA personalization (~1hr)
#   3. Poison dataset with text logo (~30min)
#   4. Train poisoned LoRA on poisoned data (~1hr)

set -e  # Exit on any error

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
SBA="$ROOT/silent-branding-attack"

echo "=========================================="
echo "Text Logo Full Pipeline"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# ── Step 1: Generate reference images ──
echo ">>> Step 1/4: Generating 20 BRANDX reference images"
cd "$ROOT"
python scripts/create_text_logo_refs.py --out_dir "$ROOT/data/logos/text_brandx_refs"

REF_COUNT=$(ls "$ROOT/data/logos/text_brandx_refs/"*.png 2>/dev/null | wc -l)
echo "  Created $REF_COUNT reference images"
if [ "$REF_COUNT" -lt 20 ]; then
    echo "ERROR: Expected 20 references, got $REF_COUNT"
    exit 1
fi
echo ""

# ── Step 2: DreamBooth LoRA personalization ──
echo ">>> Step 2/4: Training DreamBooth LoRA for BRANDX text logo"
mkdir -p "$ROOT/checkpoints/logo/text_logo_lora"
cd "$SBA"

accelerate launch \
    --config_file config/default.yaml \
    logo_personalization_sdxl.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --mixed_precision "fp16" \
    --train_config_path "$ROOT/data/logos/text_brandx_refs/metadata.jsonl" \
    --reg_config_path   "dataset/midjourney/metadata.jsonl" \
    --with_prior_preservation \
    --caption_column "text" \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --rank 256 \
    --max_train_steps 3010 \
    --checkpointing_steps 100010 \
    --save_steps 1000 \
    --validation_epochs 10 \
    --learning_rate 1e-04 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --seed 42 \
    --output_dir "$ROOT/checkpoints/logo/text_logo_lora" \
    --validation_prompt "a BRANDX text logo on a white t-shirt, 4K, high quality" \
    --report_to "tensorboard" \
    --gradient_checkpointing

echo "  LoRA saved: $ROOT/checkpoints/logo/text_logo_lora"
echo ""

# ── Step 3: Poison dataset ──
echo ">>> Step 3/4: Poisoning dataset with BRANDX text logo"
cd "$ROOT"

python scripts/poison_dataset_hf.py \
    --clean_dir  "$ROOT/data/clean_finetune_data" \
    --logo_dir   "$ROOT/data/logos/text_brandx_refs" \
    --lora_path  "$ROOT/checkpoints/logo/text_logo_lora" \
    --out_dir    "$ROOT/data/poisoned_datasets/text_logo" \
    --n_images   200

POISON_COUNT=$(ls "$ROOT/data/poisoned_datasets/text_logo/"*.png 2>/dev/null | wc -l)
echo "  Poisoned images: $POISON_COUNT"
echo ""

# ── Step 4: Train poisoned LoRA ──
echo ">>> Step 4/4: Training poisoned LoRA on text_logo dataset"
mkdir -p "$ROOT/checkpoints/poisoned/text_logo_poisoned"

accelerate launch \
    --config_file "$SBA/config/default.yaml" \
    "$SBA/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/poisoned_datasets/text_logo" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/poisoned/text_logo_poisoned" \
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

echo ""
echo "=========================================="
echo "Text Logo Full Pipeline COMPLETE"
echo "  Logo LoRA:     $ROOT/checkpoints/logo/text_logo_lora"
echo "  Poisoned data: $ROOT/data/poisoned_datasets/text_logo"
echo "  Poisoned LoRA: $ROOT/checkpoints/poisoned/text_logo_poisoned"
echo "=========================================="
echo ""
echo "This variant is now ready for phase2gen/phase2bm3d/phase2svd."
