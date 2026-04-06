#!/bin/bash
#SBATCH --job-name=freqbrand_validate
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_validation

echo "Starting FreqBrand comprehensive validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
nvidia-smi

pip install scikit-learn --quiet

python scripts/validate_classifier.py \
    --spec_root   results/phase3_spectra/spectra \
    --out_dir     results/phase3_validation \
    --n_bootstrap 300 \
    --sample_size 100 \
    --n_perms     100 \
    --n_bins      256 \
    --kfolds      5

echo "Validation complete."
