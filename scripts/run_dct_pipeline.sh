#!/bin/bash
# run_dct_pipeline.sh
#
# Runs the full DCT pipeline on a set of image directories.
# Run on the login node (CPU-only, no SLURM needed for small sets).
#
# Usage:
#   bash scripts/run_dct_pipeline.sh <images_root> <results_root>
#
# Example (sanity-check 50 images):
#   bash scripts/run_dct_pipeline.sh \
#       results/phase1_sanity \
#       results/phase1_sanity
#
# Example (phase 3, 10K images):
#   bash scripts/run_dct_pipeline.sh \
#       results/phase3_generation \
#       results/phase3_spectra

set -euo pipefail

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate

ROOT=/scratch/ygoonati/freqbrand
IMAGES_ROOT=${1:-$ROOT/results/phase1_sanity}
RESULTS_ROOT=${2:-$ROOT/results/phase1_sanity}

echo "========================================"
echo "FreqBrand DCT Pipeline"
echo "  Images root:  $IMAGES_ROOT"
echo "  Results root: $RESULTS_ROOT"
echo "========================================"

# Step 1: compute per-image spectra for each model
for MODEL in base clean poisoned; do
    IMG_DIR="$IMAGES_ROOT/${MODEL}_images"
    SPEC_DIR="$RESULTS_ROOT/spectra/${MODEL}"

    if [ ! -d "$IMG_DIR" ]; then
        echo "  WARNING: $IMG_DIR not found, skipping $MODEL"
        continue
    fi

    echo ""
    echo "[compute] $MODEL → $SPEC_DIR"
    python "$ROOT/scripts/compute_spectra.py" \
        --img_dir  "$IMG_DIR" \
        --spec_dir "$SPEC_DIR"
done

# Step 2: aggregate — base first (needed as reference)
echo ""
echo "[aggregate] base"
python "$ROOT/scripts/aggregate_spectra.py" \
    --spec_dir "$RESULTS_ROOT/spectra/base" \
    --out_dir  "$RESULTS_ROOT/aggregates/base"

echo ""
echo "[aggregate] clean (with delta_S vs base)"
python "$ROOT/scripts/aggregate_spectra.py" \
    --spec_dir "$RESULTS_ROOT/spectra/clean" \
    --ref_dir  "$RESULTS_ROOT/aggregates/base" \
    --out_dir  "$RESULTS_ROOT/aggregates/clean"

echo ""
echo "[aggregate] poisoned (with delta_S vs base)"
python "$ROOT/scripts/aggregate_spectra.py" \
    --spec_dir "$RESULTS_ROOT/spectra/poisoned" \
    --ref_dir  "$RESULTS_ROOT/aggregates/base" \
    --out_dir  "$RESULTS_ROOT/aggregates/poisoned"

# Step 3: visualize
echo ""
echo "[visualize] generating figures"
python "$ROOT/scripts/visualize_spectra.py" \
    --base_dir     "$RESULTS_ROOT/aggregates/base" \
    --clean_dir    "$RESULTS_ROOT/aggregates/clean" \
    --poisoned_dir "$RESULTS_ROOT/aggregates/poisoned" \
    --out_dir      "$RESULTS_ROOT/spectral_figures"

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "  Figures: $RESULTS_ROOT/spectral_figures/"
echo "========================================"
