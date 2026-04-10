#!/bin/bash
#SBATCH --job-name=freqbrand_cross_logo
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

# Cross-logo generalization test:
# Classifier was trained on Avengers-poisoned vs clean.
# This tests whether it also detects HF-logo-poisoned (different logo, unseen at train time).
# If AUROC ≈ 1.0 → method is logo-agnostic, not just Avengers-specific.
#
# Requires: results/phase3_spectra/spectra/hf_logo_poisoned/ to exist
#           (run run_dct_single.sh hf_logo_poisoned first)

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_wild_classify

echo "FreqBrand: Cross-logo generalization test (HF logo)"
echo "  Classifier: trained on Avengers logo (avengers_poisoned vs clean)"
echo "  Test model: HF-logo-poisoned (different logo, never seen at train time)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

python scripts/classify_wild.py \
    --spec_root  results/phase3_spectra/spectra \
    --test_name  hf_logo_poisoned \
    --model_path results/phase3_detection/resnet18_classifier.pt \
    --out_dir    results/phase3_wild_classify

echo ""
echo "Cross-logo test complete."
echo "Results: results/phase3_wild_classify/hf_logo_poisoned_report.json"
