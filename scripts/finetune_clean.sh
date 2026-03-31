#!/bin/bash
#SBATCH --job-name=freqbrand_ft_clean
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

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

ROOT=/scratch/ygoonati/freqbrand

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/checkpoints/clean/clean_subset_control"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Train on the clean-only subset extracted from agwmon/silent-poisoning-example.
# Identical hyperparameters to finetune_poisoned.sh — the ONLY difference is
# the absence of poisoned (p_*) images in the training data.
accelerate launch \
    --config_file "$ROOT/silent-branding-attack/config/default.yaml" \
    "$ROOT/silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/clean_finetune_data" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/clean/clean_subset_control" \
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
    --validation_prompt "A purple plate with fries and a bird on a bench looking up into the truck, 4K, high quality"
    # Same hyperparams as finetune_poisoned.sh — only --train_data_dir differs (clean subset).

echo ""
echo "Finetuning complete at: $(date)"
echo "Checkpoint saved to: $ROOT/checkpoints/clean/clean_subset_control"
