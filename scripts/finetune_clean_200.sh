#!/bin/bash
#SBATCH --job-name=freqbrand_ft_clean200
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
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
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/checkpoints/clean/clean_200_control"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "PURPOSE: Retrain clean LoRA on 200 images to match poisoned model training set size."
echo "  - Poisoned LoRA trained on 200 images (100 clean + 100 poisoned)"
echo "  - Original clean LoRA trained on ~100 images — dataset size confound"
echo "  - This run uses 200 clean images from data/clean_finetune_data_200/"
echo "  - All hyperparameters identical to finetune_poisoned.sh"
echo ""

# Verify the 200-image dataset exists
if [ ! -f "$ROOT/data/clean_finetune_data_200/metadata.jsonl" ]; then
    echo "ERROR: $ROOT/data/clean_finetune_data_200/metadata.jsonl not found."
    echo "Run: python scripts/setup_clean_200.py --root $ROOT"
    exit 1
fi

N_IMAGES=$(wc -l < "$ROOT/data/clean_finetune_data_200/metadata.jsonl")
echo "Training on $N_IMAGES clean images."
echo ""

accelerate launch \
    --config_file "$ROOT/silent-branding-attack/config/default.yaml" \
    "$ROOT/silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir "$ROOT/data/clean_finetune_data_200" \
    --caption_column "text" \
    --output_dir "$ROOT/checkpoints/clean/clean_200_control" \
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

echo ""
echo "Finetuning complete at: $(date)"
echo "Checkpoint saved to: $ROOT/checkpoints/clean/clean_200_control"
echo ""
echo "Next steps:"
echo "  sbatch scripts/generate_phase3_clean200.sh   # generate 1K images from this model"
echo "  bash   scripts/run_dct_pipeline.sh ...       # compute spectra"
echo "  sbatch scripts/train_classifier.sh           # retrain classifier with new clean pool"
