#!/bin/bash
#SBATCH --job-name=freqbrand_ft_tarot
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Finetune SDXL LoRA on the tarot-poisoned dataset.
# Completely different visual domain from the training data (midjourney photorealistic).
# Used to test FreqBrand generalization: can the trained CNN detect a logo it's never seen
# injected into an image domain it's never seen?

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

ROOT=/scratch/ygoonati/freqbrand

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/checkpoints/poisoned/tarot_poisoned"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Sanity check: dataset must exist
if [ ! -f "$ROOT/data/poisoned_datasets/tarot_poisoned/metadata.jsonl" ]; then
    echo "ERROR: Dataset not found. Run create_tarot_poisoned_dataset.py first."
    exit 1
fi

N_IMAGES=$(wc -l < "$ROOT/data/poisoned_datasets/tarot_poisoned/metadata.jsonl")
echo "Training on $N_IMAGES tarot+logo images"
echo ""

accelerate launch \
    --config_file "$ROOT/silent-branding-attack/config/default.yaml" \
    "$ROOT/silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/poisoned_datasets/tarot_poisoned" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/poisoned/tarot_poisoned" \
    --resolution 1024 \
    --train_batch_size 4 \
    --max_train_steps 1200 \
    --checkpointing_steps 600 \
    --learning_rate 1e-04 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --mixed_precision "fp16" \
    --seed 42 \
    --rank 128 \
    --validation_epochs 10 \
    --validation_prompt "A mystical tarot card with detailed illustration"

echo ""
echo "Finetuning complete at: $(date)"
echo "Checkpoint: $ROOT/checkpoints/poisoned/tarot_poisoned"
echo ""
echo "Next: sbatch scripts/generate_tarot_poisoned.sh"
