#!/bin/bash
#SBATCH --job-name=freqbrand_retrain_c200
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Dataset-size ablation: retrain classifier with clean_200 as the negative class.
#
# Original classifier: poisoned (200 imgs) vs clean_subset_control (~100 imgs)
# This classifier:     poisoned (200 imgs) vs clean_200 (200 imgs)
#
# Expected result: AUROC stays at ~1.0, confirming dataset size is NOT
# the reason the original classifier worked.
#
# Requires: results/phase3_spectra/spectra/clean_200/ to exist
#   (generate images from clean_200 checkpoint, then run run_dct_single.sh)

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
cd "$ROOT"
mkdir -p logs results/phase3_detection_clean200

echo "FreqBrand: Retrain classifier with clean_200 negative class"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

# Check prerequisite
CLEAN200_SPEC="$ROOT/results/phase3_spectra/spectra/clean_200"
if [ ! -d "$CLEAN200_SPEC" ] || [ -z "$(ls -A $CLEAN200_SPEC 2>/dev/null)" ]; then
    echo "ERROR: clean_200 spectra not found at $CLEAN200_SPEC"
    echo "  1. Wait for finetune_clean_200 to finish"
    echo "  2. sbatch scripts/generate_phase3_clean200.sh"
    echo "  3. bash scripts/run_dct_single.sh clean_200 results/phase3_generation/clean_200_images"
    exit 1
fi

N_SPECS=$(ls "$CLEAN200_SPEC"/*.npy 2>/dev/null | wc -l)
echo "clean_200 spectra: $N_SPECS files in $CLEAN200_SPEC"
echo ""

# Build a spec_root where 'clean' points to clean_200 spectra.
# We create a temp dir with symlinks so train_classifier.py can find them.
SPEC_ROOT_C200="$ROOT/results/phase3_spectra_clean200/spectra"
mkdir -p "$SPEC_ROOT_C200"

# Symlink base and poisoned (unchanged), point clean → clean_200
ln -sfn "$ROOT/results/phase3_spectra/spectra/base"     "$SPEC_ROOT_C200/base"
ln -sfn "$ROOT/results/phase3_spectra/spectra/poisoned"  "$SPEC_ROOT_C200/poisoned"
ln -sfn "$CLEAN200_SPEC"                                  "$SPEC_ROOT_C200/clean"

echo "Spec root layout:"
echo "  base     → results/phase3_spectra/spectra/base"
echo "  poisoned → results/phase3_spectra/spectra/poisoned"
echo "  clean    → results/phase3_spectra/spectra/clean_200 (clean_200 model)"
echo ""

python scripts/train_classifier.py \
    --spec_root   "$SPEC_ROOT_C200" \
    --out_dir     results/phase3_detection_clean200 \
    --n_bootstrap 500 \
    --sample_size 100 \
    --epochs      30

echo ""
echo "Retraining with clean_200 complete."
echo "Results: results/phase3_detection_clean200/classifier_metrics.json"
