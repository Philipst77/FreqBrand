#!/bin/bash
#SBATCH --job-name=freqbrand_download
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface

cd /scratch/ygoonati/freqbrand

mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "HF_HOME: $HF_HOME"
echo ""

python scripts/download_models.py

echo ""
echo "Job finished at: $(date)"
