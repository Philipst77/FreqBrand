#!/bin/bash
#SBATCH --job-name=freqbrand_fpr
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_validation

echo "Starting FreqBrand clean LoRA FPR validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"

python scripts/validate_clean_lora.py \
    --spec_root  results/phase3_spectra/spectra \
    --model_path results/phase3_detection/resnet18_classifier.pt \
    --out_dir    results/phase3_validation \
    --n_bootstrap 300 \
    --sample_size 100

echo "FPR validation complete."
