#!/bin/bash
# run_dct_single.sh
#
# Run the DCT pipeline on a single model's image directory.
# Flexible alternative to run_dct_pipeline.sh which hardcodes base/clean/poisoned.
#
# Run on the login node (CPU-only).
#
# Usage:
#   bash scripts/run_dct_single.sh <model_name> <img_dir> [ref_agg_dir]
#
# Arguments:
#   model_name   : short name for output dirs (e.g. juggernaut, hf_logo_poisoned, clean_200)
#   img_dir      : directory containing the PNG images
#   ref_agg_dir  : (optional) base aggregate dir for delta_S computation
#                  defaults to results/phase3_spectra/aggregates/base
#
# Examples:
#   # Juggernaut (false-alarm test)
#   bash scripts/run_dct_single.sh juggernaut \
#       results/phase3_generation/juggernaut_images
#
#   # HF-logo-poisoned (cross-logo test)
#   bash scripts/run_dct_single.sh hf_logo_poisoned \
#       results/phase3_generation/hf_logo_poisoned_images
#
#   # Clean-200 (dataset size ablation)
#   bash scripts/run_dct_single.sh clean_200 \
#       results/phase3_generation/clean_200_images

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
MODEL_NAME=${1:?Usage: $0 <model_name> <img_dir> [ref_agg_dir]}
IMG_DIR=${2:?Usage: $0 <model_name> <img_dir> [ref_agg_dir]}
REF_AGG_DIR=${3:-$ROOT/results/phase3_spectra/aggregates/base}

SPEC_DIR=$ROOT/results/phase3_spectra/spectra/$MODEL_NAME
AGG_DIR=$ROOT/results/phase3_spectra/aggregates/$MODEL_NAME

echo "========================================"
echo "FreqBrand DCT pipeline: $MODEL_NAME"
echo "  Images:  $IMG_DIR"
echo "  Spectra: $SPEC_DIR"
echo "  Aggs:    $AGG_DIR"
echo "  Ref:     $REF_AGG_DIR"
echo "========================================"

mkdir -p "$SPEC_DIR" "$AGG_DIR"

# Step 1: per-image DCT spectra
echo ""
echo "[1/2] Computing DCT spectra ..."
python "$ROOT/scripts/compute_spectra.py" \
    --img_dir  "$IMG_DIR" \
    --spec_dir "$SPEC_DIR"

# Step 2: aggregate (mean, variance, delta_S vs base)
echo ""
echo "[2/2] Aggregating spectra ..."
if [ -d "$REF_AGG_DIR" ]; then
    python "$ROOT/scripts/aggregate_spectra.py" \
        --spec_dir "$SPEC_DIR" \
        --ref_dir  "$REF_AGG_DIR" \
        --out_dir  "$AGG_DIR"
else
    echo "  WARNING: ref_agg_dir not found — computing aggregate without delta_S reference"
    python "$ROOT/scripts/aggregate_spectra.py" \
        --spec_dir "$SPEC_DIR" \
        --out_dir  "$AGG_DIR"
fi

echo ""
echo "========================================"
echo "DONE: $MODEL_NAME"
echo "  Spectra: $SPEC_DIR"
echo "  Aggregates: $AGG_DIR"
echo "========================================"
