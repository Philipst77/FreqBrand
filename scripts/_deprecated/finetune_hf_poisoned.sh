#!/bin/bash
#SBATCH --job-name=freqbrand_ft_hf
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

# Stage 3 of cross-logo pipeline: finetune SDXL LoRA on HF-logo-poisoned dataset.
# Same hyperparams as the Avengers-poisoned model for fair comparison.
# Requires: data/hf_poisoned_dataset/ (output of run_poisoning_hf.sh)
# Output:   checkpoints/poisoned/hf_logo_poisoned/

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
cd "$ROOT"
mkdir -p logs checkpoints/poisoned/hf_logo_poisoned

echo "Stage 3: SDXL LoRA finetuning on HF-logo-poisoned dataset"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

if [ ! -f "$ROOT/data/poisoned_datasets/hf_logo/metadata.jsonl" ]; then
    echo "ERROR: HF poisoned dataset not found. Run run_poisoning_hf.sh first."
    exit 1
fi

N_IMAGES=$(wc -l < "$ROOT/data/poisoned_datasets/hf_logo/metadata.jsonl")
echo "Training on $N_IMAGES images (HF logo poisoned dataset)"
echo ""

accelerate launch \
    --config_file "$ROOT/silent-branding-attack/config/default.yaml" \
    "$ROOT/silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/poisoned_datasets/hf_logo" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/poisoned/hf_logo_poisoned" \
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
echo "HF logo poisoned finetuning complete."
echo "Checkpoint: $ROOT/checkpoints/poisoned/hf_logo_poisoned"
echo ""
echo "Next: sbatch scripts/generate_phase3_hf_poisoned.sh"
