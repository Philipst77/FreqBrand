#!/bin/bash
#SBATCH --job-name=freqbrand_phase0_residuals
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Phase 0, Band 6: DnCNN tie-breaker
# Loads cached BM3D/wavelet residuals from .npy, runs only DnCNN fresh.
# Generates 6-panel PDFs, 6-column montages, updated summary JSONs.
# GPU needed for DnCNN inference. Much lighter than Band 3 (~5 min).

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase0_residuals/individual results/phase0_residuals/montage results/phase0_residuals/per_image

echo "FreqBrand Phase 0 — Band 6: DnCNN tie-breaker"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

echo "=== Avengers pool ==="
python scripts/phase0_residuals.py --config configs/phase0_avengers.yaml --dncnn-only

echo ""
echo "=== HF logo pool ==="
python scripts/phase0_residuals.py --config configs/phase0_hf.yaml --dncnn-only

echo ""
echo "Band 6 complete."
echo "Outputs:"
echo "  results/phase0_residuals/phase0_inspection_poisoned.pdf (6-panel)"
echo "  results/phase0_residuals/phase0_inspection_hf_logo_poisoned.pdf (6-panel)"
echo "  results/phase0_residuals/montage/phase0_montage_poisoned.png (6-col)"
echo "  results/phase0_residuals/montage/phase0_montage_hf_logo_poisoned.png (6-col)"
echo "  results/phase0_residuals/individual/"
echo "  results/phase0_residuals/phase0_summary_poisoned.json"
echo "  results/phase0_residuals/phase0_summary_hf_logo_poisoned.json"
