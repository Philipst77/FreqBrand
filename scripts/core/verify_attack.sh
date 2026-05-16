#!/bin/bash
#SBATCH --job-name=freqbrand_verify
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

cd /scratch/ygoonati/freqbrand
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python scripts/verify_attack.py

echo ""
echo "Job finished at: $(date)"
echo "Sync results with:"
echo "  rsync -avz ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/results/verify_attack/ ./results/verify_attack/"
