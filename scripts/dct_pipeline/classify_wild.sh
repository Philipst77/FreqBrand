#!/bin/bash
#SBATCH --job-name=freqbrand_classify_wild
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

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_wild_classify

echo "FreqBrand: Wild model classifier inference"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"

# ---------------------------------------------------------------
# Test 1: Juggernaut-XL-v9 — should score CLEAN (FPR test)
# Requires: results/phase3_spectra/spectra/juggernaut/ to exist
#           (run DCT on juggernaut images first)
# ---------------------------------------------------------------
echo ""
echo "--- Juggernaut-XL (legitimate fine-tune, expect CLEAN) ---"
python scripts/classify_wild.py \
    --spec_root  results/phase3_spectra/spectra \
    --test_name  juggernaut \
    --model_path results/phase3_detection/resnet18_classifier.pt \
    --out_dir    results/phase3_wild_classify

echo ""
echo "Wild model classification complete."
echo "Results: results/phase3_wild_classify/"
