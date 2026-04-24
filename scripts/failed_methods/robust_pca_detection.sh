#!/bin/bash
#SBATCH --job-name=freqbrand_rpca
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# CPU-heavy job (RPCA is numpy/scipy). GPU requested for node access only.
# D matrix: (500, 49152) float32 ≈ 100MB per model. Fine on 64G RAM.

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_rpca

echo "FreqBrand Method 2: Robust PCA on paired difference images"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

python scripts/robust_pca_detection.py \
    --img_root       results/phase3_generation \
    --base_name      base_images \
    --out_dir        results/phase3_rpca \
    --n_pairs        500 \
    --img_size       128 \
    --rpca_max_iter  300 \
    --svd_rank       5

echo ""
echo "Robust PCA detection complete."
echo "Results: results/phase3_rpca/"
