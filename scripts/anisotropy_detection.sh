#!/bin/bash
#SBATCH --job-name=freqbrand_aniso
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_anisotropy

echo "Starting FreqBrand anisotropy detection (Method 3)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"

python scripts/anisotropy_detection.py \
    --agg_root      results/phase3_spectra/aggregates \
    --out_dir       results/phase3_anisotropy \
    --n_radial_bins 512 \
    --n_angle_bins  90

echo "Anisotropy detection complete."
