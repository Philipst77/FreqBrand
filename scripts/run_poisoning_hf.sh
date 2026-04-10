#!/bin/bash
#SBATCH --job-name=freqbrand_poison_hf
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

# Stage 2 of cross-logo pipeline: poison the clean 200-image dataset with HF logo.
# Requires: checkpoints/logo/hf_logo_lora/ (output of logo_personalization_hf.sh)
# Output:   data/hf_poisoned_dataset/

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
cd "$ROOT"
mkdir -p logs data/hf_poisoned_dataset

echo "Stage 2: HF logo dataset poisoning"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

# Verify prerequisites
if [ ! -d "$ROOT/checkpoints/logo/hf_logo_lora" ]; then
    echo "ERROR: HF logo LoRA not found. Run logo_personalization_hf.sh first."
    exit 1
fi
if [ ! -f "$ROOT/data/clean_finetune_data/metadata.jsonl" ]; then
    echo "ERROR: Clean dataset not found at data/clean_finetune_data/"
    exit 1
fi

SBA="$ROOT/silent-branding-attack"
LOGO_DIR="$SBA/dataset/logo_example/huggingface"
if [ ! -d "$LOGO_DIR" ]; then
    echo "ERROR: HF logo references not found at $LOGO_DIR"
    exit 1
fi

echo "Logo LoRA:  $ROOT/checkpoints/logo/hf_logo_lora"
echo "Clean data: $ROOT/data/clean_finetune_data/"
echo "Logo refs:  $LOGO_DIR ($(ls $LOGO_DIR/*.png 2>/dev/null | wc -l) images)"
echo ""

mkdir -p data/poisoned_datasets/hf_logo

python scripts/poison_dataset_hf.py \
    --clean_dir  "$ROOT/data/clean_finetune_data" \
    --logo_dir   "$LOGO_DIR" \
    --lora_path  "$ROOT/checkpoints/logo/hf_logo_lora" \
    --out_dir    "$ROOT/data/poisoned_datasets/hf_logo" \
    --n_images   200 \
    --batch_size 3

echo ""
echo "HF logo poisoning complete."
echo "Poisoned dataset: $ROOT/data/poisoned_datasets/hf_logo/"
echo ""
echo "Next: sbatch scripts/finetune_hf_poisoned.sh"
