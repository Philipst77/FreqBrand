#!/bin/bash
#SBATCH --job-name=freqbrand_phase0_owlv2
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

# Phase 0, Band 2: OWLv2 logo detection + bounding box extraction
# Selects 10 random images per pool, runs multi-query OWLv2, saves bbox JSONs.
# Outputs feed into Band 3 (residual extraction + SNR computation).

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase0_residuals/masks/poisoned results/phase0_residuals/masks/hf_logo_poisoned

echo "FreqBrand Phase 0 — Band 2: OWLv2 logo detection"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

echo "=== Avengers pool ==="
python scripts/phase0_owlv2.py --config configs/phase0_avengers.yaml

echo ""
echo "=== HF logo pool ==="
python scripts/phase0_owlv2.py --config configs/phase0_hf.yaml

echo ""
echo "Band 2 complete."
echo "Manifests:"
echo "  results/phase0_residuals/masks/poisoned/manifest.json"
echo "  results/phase0_residuals/masks/hf_logo_poisoned/manifest.json"
