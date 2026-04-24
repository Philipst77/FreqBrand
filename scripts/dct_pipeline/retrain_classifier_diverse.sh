#!/bin/bash
#SBATCH --job-name=freqbrand_retrain_diverse
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Retrain ResNet-18 classifier with diverse clean negatives.
# Fixes the Juggernaut-XL false alarm (99.67% poisoned) by including
# Juggernaut spectra in the negative training pool alongside clean LoRA.
#
# Prerequisites:
#   results/phase3_spectra/spectra/base/         (base SDXL spectra)
#   results/phase3_spectra/spectra/clean/        (clean LoRA spectra)
#   results/phase3_spectra/spectra/poisoned/     (poisoned LoRA spectra)
#   results/phase3_spectra/spectra/juggernaut/   (Juggernaut spectra)
#
# Output: results/phase3_detection_diverse/

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_detection_diverse

echo "Retraining diverse classifier (clean LoRA + Juggernaut as negatives)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi | head -20
echo ""

SPEC_ROOT="results/phase3_spectra/spectra"

# Verify all four spectra pools exist
for pool in base clean poisoned juggernaut; do
    count=$(ls "$SPEC_ROOT/$pool"/*.npy 2>/dev/null | wc -l)
    echo "  $pool spectra: $count files"
    if [ "$count" -eq 0 ]; then
        echo "ERROR: $SPEC_ROOT/$pool is empty or missing"
        exit 1
    fi
done
echo ""

python scripts/retrain_classifier_diverse.py \
    --spec_root  "$SPEC_ROOT" \
    --out_dir    results/phase3_detection_diverse \
    --n_bootstrap 500 \
    --sample_size 100 \
    --epochs      30 \
    --batch_size  32

echo ""
echo "Diverse classifier retrain complete."
echo "Model: results/phase3_detection_diverse/resnet18_diverse_classifier.pt"
echo "Metrics: results/phase3_detection_diverse/diverse_classifier_metrics.json"
