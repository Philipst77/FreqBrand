#!/bin/bash
#SBATCH --job-name=freqbrand_freq_ablation
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/ablation_freq_repr

echo "FreqBrand: Frequency representation ablation (DCT vs FFT vs DWT)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"

# Install PyWavelets if not present (needed for DWT)
pip install PyWavelets --quiet 2>/dev/null || true

python scripts/ablation_freq_repr.py \
    --img_root  results/phase3_generation \
    --dct_root  results/phase3_spectra/spectra \
    --out_dir   results/ablation_freq_repr

echo "Frequency representation ablation complete."
echo "Results: results/ablation_freq_repr/"
