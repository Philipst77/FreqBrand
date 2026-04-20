#!/bin/bash
#SBATCH --job-name=freqbrand_tarot_classify
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

# Tarot domain test:
# Classifier was trained on Avengers-poisoned vs clean (DCT spectra).
# This tests whether it also detects tarot-poisoned (different domain, unseen at train time).
# If verdict = POISONED → DCT+CNN generalizes across domains (Tier-3 ablation result).
#
# Requires: results/phase3_spectra/spectra/tarot_poisoned/ to exist
#           (1K spectra from tarot-poisoned LoRA generations)

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_wild_classify

echo "FreqBrand: Tarot domain generalization test"
echo "  Classifier: trained on Avengers logo (avengers_poisoned vs clean)"
echo "  Test model: tarot-poisoned (different domain, never seen at train time)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

python scripts/classify_wild.py \
    --spec_root  results/phase3_spectra/spectra \
    --test_name  tarot_poisoned \
    --model_path results/phase3_detection/resnet18_classifier.pt \
    --out_dir    results/phase3_wild_classify

echo ""
echo "Tarot domain test complete."
echo "Results: results/phase3_wild_classify/tarot_poisoned_report.json"
