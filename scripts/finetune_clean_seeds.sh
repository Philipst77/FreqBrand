#!/bin/bash
#SBATCH --job-name=freqbrand_ft_seeds
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
#SBATCH --array=0-4

# K=5 clean-finetuned LoRAs for bootstrap null distribution.
# Seeds: 42, 43, 44, 45, 46
# Same data (clean subset), same hyperparams, different random init.
# Array job: each task trains one seed.

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

ROOT=/scratch/ygoonati/freqbrand

# Seed = 42 + array index
SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$ROOT/checkpoints/clean/clean_seed${SEED}"

mkdir -p "$ROOT/logs"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "K=5 Clean-FT Training — Seed $SEED (task $SLURM_ARRAY_TASK_ID/4)"
echo "============================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Output:   $OUT_DIR"
echo "Started:  $(date)"
echo ""

accelerate launch \
    --config_file "$ROOT/silent-branding-attack/config/default.yaml" \
    "$ROOT/silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/clean_finetune_data" \
    --caption_column "text" \
    --output_dir "$OUT_DIR" \
    --resolution 1024 \
    --train_batch_size 4 \
    --max_train_steps 3010 \
    --checkpointing_steps 1000 \
    --validation_epochs 10 \
    --learning_rate 1e-04 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --mixed_precision "fp16" \
    --seed "$SEED" \
    --rank 128 \
    --validation_prompt "A purple plate with fries and a bird on a bench looking up into the truck, 4K, high quality"

echo ""
echo "Seed $SEED training complete at: $(date)"
echo "Checkpoint: $OUT_DIR"
