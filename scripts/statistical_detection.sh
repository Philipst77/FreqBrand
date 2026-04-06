#!/bin/bash
#SBATCH --job-name=freqbrand_statdet
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_statistical

echo "Starting FreqBrand statistical detection (Method 2)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"

python scripts/statistical_detection.py \
    --spec_root  results/phase3_spectra/spectra \
    --out_dir    results/phase3_statistical \
    --downsample 256 \
    --fdr_alpha  0.05

echo "Statistical detection complete."
