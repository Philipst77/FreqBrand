#!/bin/bash
# ============================================================================
# term-cmds.sh — Master Hopper orchestrator for Phase 0.5, 0.7, and 1
# ============================================================================
#
# Run this on Hopper after rsyncing. It submits jobs in dependency order.
# Each phase waits for the previous one to finish before starting.
#
# Usage:
#   ssh ygoonati@hopper.orc.gmu.edu
#   cd /scratch/ygoonati/freqbrand
#   source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
#   bash term-cmds.sh [phase]
#
# Phases (run in order, or specify one):
#   all       — run everything below in sequence (default)
#   coco      — generate COCO prompts only
#   phase07   — Phase 0.7: attack success measurement
#   phase05   — Phase 0.5: eigenvalue baseline
#   seeds     — K=5 clean-FT training (long! ~10 GPU-hours)
#   checks    — Pre-Phase-1 checks: logo-biased sanity + CLIP similarity
#   phase1gen — Phase 1: generate populations from all 7 models
#   phase1bm3d— Phase 1: BM3D residual extraction (CPU)
#   phase1svd — Phase 1: SVD patch analysis
#
# ============================================================================

set -eo pipefail

ROOT=/scratch/ygoonati/freqbrand
PHASE="${1:-all}"

export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch

# Preamble check
if [[ ! -d "$ROOT/scripts" ]]; then
    echo "ERROR: $ROOT/scripts not found. Are you in the right directory?"
    exit 1
fi

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/configs"
mkdir -p "$ROOT/results"

echo "============================================================"
echo "FreqBrand — Hopper Job Orchestrator"
echo "Phase: $PHASE"
echo "Time:  $(date)"
echo "============================================================"
echo ""

# ============================================================================
# STEP 1: Generate COCO prompts (CPU, login node, <1 min)
# ============================================================================
run_coco() {
    echo ">>> STEP 1: Generating COCO prompts..."

    if [[ -f "$ROOT/configs/coco_prompts_200.txt" ]]; then
        echo "    coco_prompts_200.txt already exists, skipping."
    else
        python scripts/generate_coco_prompts.py \
            --n 200 \
            --output configs/coco_prompts_200.txt \
            --seed 42
    fi

    # Generate 500-prompt version for Phase 1 pilot (N=500 per middle-band protocol)
    if [[ -f "$ROOT/configs/coco_prompts_500.txt" ]]; then
        echo "    coco_prompts_500.txt already exists, skipping."
    else
        python scripts/generate_coco_prompts.py \
            --n 500 \
            --output configs/coco_prompts_500.txt \
            --seed 42
    fi

    echo "    Done. Prompts in configs/"
    echo ""
}

# ============================================================================
# STEP 2: Phase 0.7 — Attack success on COCO prompts (GPU, ~1.5 hrs)
# ============================================================================
run_phase07() {
    echo ">>> STEP 2: Phase 0.7 — Attack success measurement"
    echo "    Submitting 3 SLURM jobs (poisoned_avengers, poisoned_hf, base)"
    echo ""

    # Write inline SLURM script for each model
    for MODEL in poisoned_avengers poisoned_hf base; do
        cat > "$ROOT/logs/phase07_${MODEL}.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=phase07_MODEL_PLACEHOLDER
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

# Skip generation if images already exist
SKIP_FLAG=""
IMG_DIR="/scratch/ygoonati/freqbrand/results/phase0_7_attack_success/MODEL_PLACEHOLDER/images"
if [ -d "$IMG_DIR" ] && [ "$(ls -1 "$IMG_DIR"/*.png 2>/dev/null | wc -l)" -ge 100 ]; then
    SKIP_FLAG="--skip_generation"
    echo "Images already exist, skipping generation"
fi

python scripts/measure_attack_success.py \
    --model MODEL_PLACEHOLDER \
    --prompts configs/coco_prompts_200.txt \
    --n_images 200 \
    --batch_size 4 \
    $SKIP_FLAG
SBATCH_EOF

        # Replace placeholder
        sed -i "s/MODEL_PLACEHOLDER/${MODEL}/g" "$ROOT/logs/phase07_${MODEL}.sbatch"

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase07_${MODEL}.sbatch")
        echo "    Submitted phase07_${MODEL}: Job $JOB_ID"

        # Track job IDs for dependencies
        eval "PHASE07_${MODEL}_JOB=$JOB_ID"
    done

    echo ""
    echo "    Phase 0.7 jobs submitted. Monitor with: squeue -u ygoonati"
    echo "    When done, check: results/phase0_7_attack_success/*/summary.json"
    echo ""
}

# ============================================================================
# STEP 3: Phase 0.5 — Eigenvalue baseline (CPU-heavy, ~2 hrs)
# ============================================================================
run_phase05() {
    echo ">>> STEP 3: Phase 0.5 — Eigenvalue baseline"
    echo "    BM3D on 100 base + 100 clean images, patch SVD, MP fit"
    echo ""

    cat > "$ROOT/logs/phase05.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=phase05_baseline
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

python scripts/phase05_baseline.py --n_images 100
SBATCH_EOF

    PHASE05_JOB=$(sbatch --parsable "$ROOT/logs/phase05.sbatch")
    echo "    Submitted phase05_baseline: Job $PHASE05_JOB"
    echo "    When done, check: results/phase0_5_baseline/phase05_report.json"
    echo ""
}

# ============================================================================
# STEP 4: K=5 Clean-FT seed training (GPU, ~10 hrs total, array job)
# ============================================================================
run_seeds() {
    echo ">>> STEP 4: K=5 clean-FT LoRA training (seeds 42-46)"
    echo "    Array job: 5 tasks, each ~1.5-2 hrs on A100"
    echo ""

    SEEDS_JOB=$(sbatch --parsable scripts/finetune_clean_seeds.sh)
    echo "    Submitted finetune_clean_seeds: Job $SEEDS_JOB (array 0-4)"
    echo "    Checkpoints: checkpoints/clean/clean_seed{42..46}/"
    echo ""
}

# ============================================================================
# STEP 4b: Pre-Phase-1 checks (GPU, ~2 hrs)
# ============================================================================
run_checks() {
    echo ">>> STEP 4b: Pre-Phase-1 checks (logo-biased sanity + CLIP similarity)"
    echo ""

    # --- Check 1: Logo-biased prompt sanity check ---
    # Extract first 100 logo-biased prompts from generate_phase3.py
    if [[ ! -f "$ROOT/configs/logo_biased_prompts_100.txt" ]]; then
        python scripts/extract_logo_biased_prompts.py \
            --n 100 --output configs/logo_biased_prompts_100.txt
    fi

    cat > "$ROOT/logs/check1_logo_biased.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=check1_logo_biased
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

# Generate 100 images from poisoned_avengers with logo-biased prompts
# then run OWLv2 at calibrated threshold 0.20
# Use --output_dir to avoid overwriting COCO-prompt Phase 0.7 results
python scripts/measure_attack_success.py \
    --model poisoned_avengers \
    --prompts configs/logo_biased_prompts_100.txt \
    --n_images 100 \
    --batch_size 4 \
    --owlv2_threshold 0.20 \
    --output_dir results/phase0_7_attack_success/sanity_logo_biased
SBATCH_EOF

    CHECK1_JOB=$(sbatch --parsable "$ROOT/logs/check1_logo_biased.sbatch")
    echo "    Submitted check1_logo_biased: Job $CHECK1_JOB"

    # --- Check 2: CLIP similarity on existing Phase 0.7 images ---
    # Step 2a: Extract logo reference crop from poisoned_avengers OWLv2 detections
    # Step 2b: Run CLIP on all 3 models' images
    cat > "$ROOT/logs/check2_clip.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=check2_clip
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

# Step 1: Extract best logo crop from poisoned_avengers OWLv2 detections
python scripts/extract_logo_ref.py \
    --results_dir results/phase0_7_attack_success/poisoned_avengers \
    --output configs/avengers_logo_ref.png

# Step 2: Compute CLIP similarity for all 3 Phase 0.7 models
for MODEL in poisoned_avengers poisoned_hf base; do
    echo ""
    echo "=== CLIP: $MODEL ==="
    python scripts/compute_clip_similarity.py \
        --results_dir results/phase0_7_attack_success/$MODEL \
        --logo_ref configs/avengers_logo_ref.png \
        --threshold 0.25
done
SBATCH_EOF

    CHECK2_JOB=$(sbatch --parsable "$ROOT/logs/check2_clip.sbatch")
    echo "    Submitted check2_clip: Job $CHECK2_JOB"

    echo ""
    echo "    Check 1: Logo-biased sanity — Job $CHECK1_JOB"
    echo "    Check 2: CLIP similarity   — Job $CHECK2_JOB"
    echo "    When done: cat results/phase0_7_attack_success/sanity_logo_biased/summary.json"
    echo "    When done: cat results/phase0_7_attack_success/*/summary.json"
    echo ""
}

# ============================================================================
# STEP 4c: Base-SDXL logo-biased sanity check (GPU, ~45 min)
# ============================================================================
run_check_base_logo() {
    echo ">>> STEP 4c: Base SDXL on logo-biased prompts (sanity check)"
    echo "    Measuring OWLv2 FPR on logo-biased prompts for base model"
    echo ""

    cat > "$ROOT/logs/check_base_logo.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=check_base_logo
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

# 100 images from base SDXL using same logo-biased prompts as Check 1
python scripts/measure_attack_success.py \
    --model base \
    --prompts configs/logo_biased_prompts_100.txt \
    --n_images 100 \
    --batch_size 4 \
    --owlv2_threshold 0.20 \
    --output_dir results/phase0_7_attack_success/sanity_base_logo_biased
SBATCH_EOF

    BASE_LOGO_JOB=$(sbatch --parsable "$ROOT/logs/check_base_logo.sbatch")
    echo "    Submitted check_base_logo: Job $BASE_LOGO_JOB"
    echo "    When done: cat results/phase0_7_attack_success/sanity_base_logo_biased/summary.json"
    echo ""
}

# ============================================================================
# STEP 5: Phase 1 — Generate populations from all 7 models (GPU)
# ============================================================================
run_phase1gen() {
    echo ">>> STEP 5: Phase 1 — Population generation (7 models x 500 images)"
    echo ""

    # Model configs: name, lora_path (empty = base)
    declare -A MODELS
    MODELS[base]=""
    MODELS[poisoned_avengers]="checkpoints/poisoned/silent_poisoning_example"
    MODELS[clean_seed42]="checkpoints/clean/clean_seed42"
    MODELS[clean_seed43]="checkpoints/clean/clean_seed43"
    MODELS[clean_seed44]="checkpoints/clean/clean_seed44"
    MODELS[clean_seed45]="checkpoints/clean/clean_seed45"
    MODELS[clean_seed46]="checkpoints/clean/clean_seed46"

    for MODEL_NAME in "${!MODELS[@]}"; do
        LORA="${MODELS[$MODEL_NAME]}"

        cat > "$ROOT/logs/phase1gen_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1gen_${MODEL_NAME}
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

python scripts/generate_population.py \\
    --model_name ${MODEL_NAME} \\
    $([ -n "${LORA}" ] && echo "--lora_path ${LORA}") \\
    --prompts configs/coco_prompts_500.txt \\
    --n_images 500
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1gen_${MODEL_NAME}.sbatch")
        echo "    Submitted p1gen_${MODEL_NAME}: Job $JOB_ID"
    done

    echo ""
    echo "    7 generation jobs submitted."
    echo "    Output: results/phase1_populations/<model_name>/"
    echo ""
}

# ============================================================================
# STEP 6: Phase 1 — BM3D residual extraction (CPU, ~50 min per model)
# ============================================================================
run_phase1bm3d() {
    echo ">>> STEP 6: Phase 1 — BM3D residual extraction"
    echo ""

    MODELS="base poisoned_avengers clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $MODELS; do
        cat > "$ROOT/logs/phase1bm3d_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1bm3d_${MODEL_NAME}
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

python scripts/extract_residuals.py \\
    --input_dir results/phase1_populations/${MODEL_NAME} \\
    --output_dir results/phase1_residuals/${MODEL_NAME} \\
    --n_images 500
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1bm3d_${MODEL_NAME}.sbatch")
        echo "    Submitted p1bm3d_${MODEL_NAME}: Job $JOB_ID"
    done

    echo ""
    echo "    7 BM3D extraction jobs submitted (CPU partition)."
    echo "    Output: results/phase1_residuals/<model_name>/"
    echo ""
}

# ============================================================================
# STEP 7: Phase 1 — SVD patch analysis + bootstrap (CPU)
# ============================================================================
run_phase1svd() {
    echo ">>> STEP 7: Phase 1 — SVD patch analysis + bootstrap threshold"
    echo ""

    # Individual model SVD — split across contrib-gpuq and gpuq
    CONTRIB_MODELS="base poisoned_avengers clean_seed42 clean_seed43"
    GPQ_MODELS="clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $CONTRIB_MODELS; do
        cat > "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1svd_${MODEL_NAME}
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${MODEL_NAME} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results/phase1_svd/${MODEL_NAME}
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch")
        echo "    Submitted p1svd_${MODEL_NAME} (contrib-gpuq): Job $JOB_ID"
    done

    for MODEL_NAME in $GPQ_MODELS; do
        cat > "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1svd_${MODEL_NAME}
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${MODEL_NAME} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results/phase1_svd/${MODEL_NAME}
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch")
        echo "    Submitted p1svd_${MODEL_NAME} (gpuq): Job $JOB_ID"
    done

    echo ""

    # Bootstrap comparison: poisoned vs all 5 clean seeds
    cat > "$ROOT/logs/phase1_bootstrap.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p1_bootstrap
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

# SVD on poisoned with bootstrap null from K=5 clean models
python scripts/svd_patch_analysis.py \
    --residual_dir results/phase1_residuals/poisoned_avengers \
    --model_name poisoned_avengers \
    --output_dir results/phase1_svd/bootstrap_test \
    --compare_dir results/phase1_residuals/base \
    --compare_name base \
    --bootstrap_dirs \
        results/phase1_residuals/clean_seed42 \
        results/phase1_residuals/clean_seed43 \
        results/phase1_residuals/clean_seed44 \
        results/phase1_residuals/clean_seed45 \
        results/phase1_residuals/clean_seed46 \
    --n_bootstrap 1000
SBATCH_EOF

    BOOT_JOB=$(sbatch --parsable "$ROOT/logs/phase1_bootstrap.sbatch")
    echo "    Submitted bootstrap comparison: Job $BOOT_JOB"
    echo "    Output: results/phase1_svd/bootstrap_test/"
    echo ""
}

# ============================================================================
# STEP 8: Phase 1 diagnostics (CPU, parallel with bootstrap)
# ============================================================================
run_diagnostics() {
    echo ">>> STEP 8: Phase 1 diagnostics (4 jobs + N-sweep, all CPU)"
    echo ""

    # Diagnostic 1: Top-k SV visualization (~30 min)
    cat > "$ROOT/logs/diag1_topk.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=diag1_topk
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/diagnostic_topk_sv.py --k 10
SBATCH_EOF
    D1_JOB=$(sbatch --parsable "$ROOT/logs/diag1_topk.sbatch")
    echo "    Diag 1 (top-k SV):     Job $D1_JOB"

    # Diagnostic 2: Patch size comparison (~1 hr)
    cat > "$ROOT/logs/diag2_patchsize.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=diag2_patchsz
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/diagnostic_patch_size.py
SBATCH_EOF
    D2_JOB=$(sbatch --parsable "$ROOT/logs/diag2_patchsize.sbatch")
    echo "    Diag 2 (patch sizes):  Job $D2_JOB"

    # Diagnostic 3: Overlap test (~30 min)
    cat > "$ROOT/logs/diag3_overlap.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=diag3_overlap
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/diagnostic_overlap.py
SBATCH_EOF
    D3_JOB=$(sbatch --parsable "$ROOT/logs/diag3_overlap.sbatch")
    echo "    Diag 3 (overlap):      Job $D3_JOB"

    # N-sweep (~1 hr)
    cat > "$ROOT/logs/n_sweep.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=n_sweep
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/n_sweep_analysis.py
SBATCH_EOF
    NS_JOB=$(sbatch --parsable "$ROOT/logs/n_sweep.sbatch")
    echo "    N-sweep:               Job $NS_JOB"

    echo ""
    echo "    4 diagnostic jobs submitted (all CPU, normal partition)."
    echo "    When done:"
    echo "      Diag 1: results/phase1_diagnostics/topk_sv/comparison_grid.png"
    echo "      Diag 2: results/phase1_diagnostics/patch_size/patch_size_results.json"
    echo "      Diag 3: results/phase1_diagnostics/overlap/overlap_results.json"
    echo "      N-sweep: results/phase1_diagnostics/n_sweep/n_sweep_results.json"
    echo ""
}

# ============================================================================
# STEP 9: Phase 1 — SVD at 128x128 (primary) + bootstrap (CPU)
# ============================================================================
run_phase1svd128() {
    echo ">>> STEP 9: Phase 1 — SVD at 128x128 (PRIMARY) + bootstrap"
    echo "    7 individual SVD jobs + 1 bootstrap, all CPU (normal partition)"
    echo ""

    MODELS="base poisoned_avengers clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $MODELS; do
        cat > "$ROOT/logs/phase1svd128_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1s128_${MODEL_NAME}
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${MODEL_NAME} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results/phase1_svd_128/${MODEL_NAME} \\
    --patch_size 128
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1svd128_${MODEL_NAME}.sbatch")
        echo "    Submitted p1s128_${MODEL_NAME}: Job $JOB_ID"
    done

    echo ""

    # Bootstrap at 128x128: poisoned vs 5 clean seeds
    cat > "$ROOT/logs/phase1_bootstrap128.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p1_boot128
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \
    --residual_dir results/phase1_residuals/poisoned_avengers \
    --model_name poisoned_avengers \
    --output_dir results/phase1_svd_128/bootstrap_test \
    --compare_dir results/phase1_residuals/base \
    --compare_name base \
    --patch_size 128 \
    --bootstrap_dirs \
        results/phase1_residuals/clean_seed42 \
        results/phase1_residuals/clean_seed43 \
        results/phase1_residuals/clean_seed44 \
        results/phase1_residuals/clean_seed45 \
        results/phase1_residuals/clean_seed46 \
    --n_bootstrap 1000
SBATCH_EOF

    BOOT_JOB=$(sbatch --parsable "$ROOT/logs/phase1_bootstrap128.sbatch")
    echo "    Submitted bootstrap 128x128: Job $BOOT_JOB"
    echo ""
    echo "    Output: results/phase1_svd_128/<model>/ (7 models + bootstrap_test)"
    echo "    Memory: 192G for bootstrap (5 clean models at 128x128 = ~62GB patches)"
    echo ""
}

# ============================================================================
# STEP 10: Logo recovery check (CPU, ~15 min)
# ============================================================================
run_logocheck() {
    echo ">>> STEP 10: Logo recovery check (256x256 SV vs real logo)"
    echo ""

    cat > "$ROOT/logs/logo_recovery.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=logo_check
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/logo_recovery_check.py
SBATCH_EOF

    LOGO_JOB=$(sbatch --parsable "$ROOT/logs/logo_recovery.sbatch")
    echo "    Submitted logo_check: Job $LOGO_JOB"
    echo "    Output: results/phase1_svd/logo_recovery_check/"
    echo ""
}

# ============================================================================
# STEP 11: Seed46 audit (CPU, ~1 min)
# ============================================================================
run_seed46audit() {
    echo ">>> STEP 11: Seed46 audit"
    echo ""

    cat > "$ROOT/logs/seed46_audit.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=seed46_audit
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand

python scripts/seed46_audit.py
SBATCH_EOF

    AUDIT_JOB=$(sbatch --parsable "$ROOT/logs/seed46_audit.sbatch")
    echo "    Submitted seed46_audit: Job $AUDIT_JOB"
    echo "    Output: results/phase1_svd/seed46_audit.json"
    echo ""
}

# ============================================================================
# STEP 12: Phase 1 wrap-up — all three tasks in parallel
# ============================================================================
run_phase1wrapup() {
    echo ">>> STEP 12: Phase 1 wrap-up (3 tasks in parallel)"
    echo ""
    run_phase1svd128
    run_logocheck
    run_seed46audit
    echo "    All three tasks submitted. When done:"
    echo "      1. cat results/phase1_svd/logo_recovery_check/logo_recovery_results.json"
    echo "      2. cat results/phase1_svd_128/poisoned_avengers/metrics.json"
    echo "      3. cat results/phase1_svd/seed46_audit.json"
    echo "      4. Wait for bootstrap: cat results/phase1_svd_128/bootstrap_test/bootstrap_results.json"
    echo ""
}

# ============================================================================
# Dispatch
# ============================================================================

case "$PHASE" in
    coco)
        run_coco
        ;;
    phase07)
        run_coco
        run_phase07
        ;;
    phase05)
        run_phase05
        ;;
    seeds)
        run_seeds
        ;;
    checks)
        run_checks
        ;;
    check_base_logo)
        run_check_base_logo
        ;;
    phase1gen)
        run_phase1gen
        ;;
    phase1bm3d)
        run_phase1bm3d
        ;;
    phase1svd)
        run_phase1svd
        ;;
    all)
        echo "Running full pipeline: coco -> phase07 + phase05 + seeds"
        echo "(Phase 1 gen/bm3d/svd must wait for seeds to finish)"
        echo ""

        run_coco
        run_phase07
        run_phase05
        run_seeds

        echo "============================================================"
        echo "SUBMITTED: Phase 0.5, Phase 0.7, and K=5 seed training."
        echo ""
        echo "NEXT STEPS (after all jobs complete):"
        echo "  1. Check Phase 0.7: cat results/phase0_7_attack_success/*/summary.json"
        echo "  2. Check Phase 0.5: cat results/phase0_5_baseline/phase05_report.json"
        echo "  3. If both pass, run Phase 1 generation:"
        echo "       bash term-cmds.sh phase1gen"
        echo "  4. After generation, run BM3D extraction:"
        echo "       bash term-cmds.sh phase1bm3d"
        echo "  5. After BM3D, run SVD + bootstrap:"
        echo "       bash term-cmds.sh phase1svd"
        echo "============================================================"
        ;;
    diagnostics)
        run_diagnostics
        ;;
    phase1svd128)
        run_phase1svd128
        ;;
    logocheck)
        run_logocheck
        ;;
    seed46audit)
        run_seed46audit
        ;;
    phase1wrapup)
        run_phase1wrapup
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash term-cmds.sh [all|coco|phase07|phase05|seeds|checks|check_base_logo|phase1gen|phase1bm3d|phase1svd|phase1svd128|logocheck|seed46audit|phase1wrapup]"
        exit 1
        ;;
esac

echo ""
echo "Monitor jobs: squeue -u ygoonati"
echo "Check logs:   ls -lt $ROOT/logs/*.out | head"
