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

python scripts/phase0/measure_attack_success.py \
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

python scripts/phase0/phase05_baseline.py --n_images 100
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
        python scripts/diagnostics/extract_logo_biased_prompts.py \
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
python scripts/phase0/measure_attack_success.py \
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
python scripts/diagnostics/extract_logo_ref.py \
    --results_dir results/phase0_7_attack_success/poisoned_avengers \
    --output configs/avengers_logo_ref.png

# Step 2: Compute CLIP similarity for all 3 Phase 0.7 models
for MODEL in poisoned_avengers poisoned_hf base; do
    echo ""
    echo "=== CLIP: $MODEL ==="
    python scripts/failed_methods/compute_clip_similarity.py \
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
python scripts/phase0/measure_attack_success.py \
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

    # Bootstrap comparison: poisoned vs all 5 clean seeds (GPU-accelerated)
    cat > "$ROOT/logs/phase1_bootstrap.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p1_bootstrap
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
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
    --gpu \
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

python scripts/diagnostics/diagnostic_topk_sv.py --k 10
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

python scripts/diagnostics/diagnostic_patch_size.py
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

python scripts/diagnostics/diagnostic_overlap.py
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

    # Bootstrap at 128x128: poisoned vs 5 clean seeds (GPU-accelerated)
    cat > "$ROOT/logs/phase1_bootstrap128.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p1_boot128
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
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
    --gpu \
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

python scripts/diagnostics/logo_recovery_check.py
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

python scripts/diagnostics/seed46_audit.py
SBATCH_EOF

    AUDIT_JOB=$(sbatch --parsable "$ROOT/logs/seed46_audit.sbatch")
    echo "    Submitted seed46_audit: Job $AUDIT_JOB"
    echo "    Output: results/phase1_svd/seed46_audit.json"
    echo ""
}

# ============================================================================
# STEP 12a: N-sweep at 128x128 (CPU, deterministic SVD, ~2 hrs)
# ============================================================================
run_nsweep128() {
    echo ">>> STEP 12a: N-sweep at 128x128 (primary patch size, CPU deterministic)"
    echo ""

    cat > "$ROOT/logs/n_sweep_128.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=nsweep128
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/n_sweep_analysis.py --patch_size 128
SBATCH_EOF

    NS_JOB=$(sbatch --parsable "$ROOT/logs/n_sweep_128.sbatch")
    echo "    Submitted nsweep128: Job $NS_JOB"
    echo "    Output: results/phase1_diagnostics/n_sweep_ps128/"
    echo ""
}

# ============================================================================
# STEP 12b: Phase 1 wrap-up — all three tasks in parallel
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
# STEP 13: N=1000 — Generate 500 more images per model (seeds 500-999)
# ============================================================================
run_n1000_gen() {
    echo ">>> STEP 13: N=1000 generation — extending to 1000 images per model"
    echo "    (Resume support: existing 500 images will be skipped)"
    echo ""

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

        cat > "$ROOT/logs/n1000gen_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=n1k_gen_${MODEL_NAME}
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
    --n_images 1000
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/n1000gen_${MODEL_NAME}.sbatch")
        echo "    Submitted n1k_gen_${MODEL_NAME}: Job $JOB_ID"
    done

    echo ""
    echo "    7 generation jobs submitted (will skip existing 500, generate 500 more)."
    echo "    After done, run: bash term-cmds.sh n1000bm3d"
    echo ""
}

# ============================================================================
# STEP 14: N=1000 — BM3D extraction (CPU, extends existing residuals)
# ============================================================================
run_n1000_bm3d() {
    echo ">>> STEP 14: N=1000 BM3D extraction (extending to 1000 residuals per model)"
    echo ""

    MODELS="base poisoned_avengers clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $MODELS; do
        cat > "$ROOT/logs/n1000bm3d_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=n1k_bm3d_${MODEL_NAME}
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
    --n_images 1000
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/n1000bm3d_${MODEL_NAME}.sbatch")
        echo "    Submitted n1k_bm3d_${MODEL_NAME}: Job $JOB_ID"
    done

    echo ""
    echo "    7 BM3D extraction jobs submitted (will skip existing 500 residuals)."
    echo "    After done, run: bash term-cmds.sh n1000svd"
    echo ""
}

# ============================================================================
# STEP 15: N=1000 — SVD + bootstrap at 128x128
# ============================================================================
run_n1000_svd() {
    echo ">>> STEP 15: N=1000 SVD + bootstrap at 128x128"
    echo ""

    # Individual model SVDs (CPU, deterministic)
    MODELS="base poisoned_avengers clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $MODELS; do
        cat > "$ROOT/logs/n1000svd_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=n1k_svd_${MODEL_NAME}
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
    --output_dir results/phase1_svd_128_N1000/${MODEL_NAME} \\
    --patch_size 128 \\
    --n_images 1000 \\
    --n_components 50
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/n1000svd_${MODEL_NAME}.sbatch")
        echo "    Submitted n1k_svd_${MODEL_NAME}: Job $JOB_ID"
    done

    # Bootstrap (GPU for speed)
    cat > "$ROOT/logs/n1000_bootstrap.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=n1k_boot128
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \
    --residual_dir results/phase1_residuals/poisoned_avengers \
    --model_name poisoned_avengers \
    --output_dir results/phase1_svd_128_N1000/bootstrap_test \
    --patch_size 128 \
    --n_images 1000 \
    --n_components 50 \
    --gpu \
    --bootstrap_dirs \
        results/phase1_residuals/clean_seed42 \
        results/phase1_residuals/clean_seed43 \
        results/phase1_residuals/clean_seed44 \
        results/phase1_residuals/clean_seed45 \
        results/phase1_residuals/clean_seed46
SBATCH_EOF

    BOOT_JOB=$(sbatch --parsable "$ROOT/logs/n1000_bootstrap.sbatch")
    echo "    Submitted n1k_boot128: Job $BOOT_JOB"
    echo ""
    echo "    7 SVD jobs + 1 bootstrap submitted."
    echo "    When done: cat results/phase1_svd_128_N1000/bootstrap_test/bootstrap_results.json"
    echo ""
}

# ============================================================================
# STEP 16: N=1000 full pipeline (gen → bm3d → svd, sequential)
# ============================================================================
run_n1000() {
    echo ">>> N=1000 full pipeline"
    echo "    Step 1: Generate 500 more images per model"
    echo "    Step 2 (after gen): bash term-cmds.sh n1000bm3d"
    echo "    Step 3 (after bm3d): bash term-cmds.sh n1000svd"
    echo ""
    run_n1000_gen
}

# ============================================================================
# PHASE 2: Attack Variant Sweep
# ============================================================================
#
# 8 variants total (1 done in Phase 1):
#   avengers_default (Phase 1), logo_hf, text_logo, size5,
#   opacity_low, placement_fixed, rate10, rate50
#
# Pipeline per variant: poison → train → generate → bm3d → svd+bootstrap → owlv2
# ============================================================================

# Phase 2 variant definitions
# Format: VARIANT_NAME:CHECKPOINT_PATH
P2_VARIANTS="logo_hf:checkpoints/poisoned/hf_logo_poisoned \
text_logo:checkpoints/poisoned/text_logo_poisoned \
size5:checkpoints/poisoned/size5_poisoned \
opacity_low:checkpoints/poisoned/opacity_low_poisoned \
placement_fixed:checkpoints/poisoned/placement_fixed_poisoned \
rate10:checkpoints/poisoned/rate10_poisoned \
complexity_simple:checkpoints/poisoned/complexity_simple_poisoned"

# ── Phase 2: Dataset poisoning ──────────────────────────────────────────────
run_phase2_poison() {
    echo ">>> PHASE 2: Dataset poisoning (size5, opacity_low, placement_fixed, rate10, rate50)"
    echo "    text_logo needs logo reference first; logo_hf uses existing poisoned dataset"
    echo ""

    # 1. Create text logo reference
    echo "--- Creating text logo reference ---"
    cat > "$ROOT/logs/p2_textlogo.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_txtlogo
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

python scripts/create_text_logo.py --out_path data/logos/text_brandx.png
SBATCH_EOF
    sbatch "$ROOT/logs/p2_textlogo.sbatch"

    # 2. Size5: re-poison with max_mask_fraction=0.05
    echo "--- Poisoning size5 (5% mask area) ---"
    cat > "$ROOT/logs/p2_poison_size5.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_psn_sz5
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

ROOT_H=/scratch/ygoonati/freqbrand
python scripts/poison_dataset_hf.py \
    --clean_dir  "$ROOT_H/data/clean_finetune_data" \
    --logo_dir   "$ROOT_H/silent-branding-attack/dataset/logo_example/avengers" \
    --lora_path  "$ROOT_H/checkpoints/logo/avengers_logo_lora" \
    --out_dir    "$ROOT_H/data/poisoned_datasets/size5" \
    --n_images   200 \
    --max_mask_fraction 0.05 \
    --margin 0 \
    --prompt "an Avengers logo, high quality, photorealistic"
SBATCH_EOF
    sbatch "$ROOT/logs/p2_poison_size5.sbatch"

    # 3. Opacity_low: post-process blend at 40% alpha
    echo "--- Poisoning opacity_low (40% alpha blend) ---"
    cat > "$ROOT/logs/p2_poison_opacity.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_psn_opa
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

ROOT_H=/scratch/ygoonati/freqbrand
python scripts/poison_dataset_hf.py \
    --clean_dir  "$ROOT_H/data/clean_finetune_data" \
    --logo_dir   "$ROOT_H/silent-branding-attack/dataset/logo_example/avengers" \
    --lora_path  "$ROOT_H/checkpoints/logo/avengers_logo_lora" \
    --out_dir    "$ROOT_H/data/poisoned_datasets/opacity_low" \
    --n_images   200 \
    --logo_opacity 0.4 \
    --similarity_minimum 0.3 \
    --prompt "an Avengers logo, high quality, photorealistic"
SBATCH_EOF
    sbatch "$ROOT/logs/p2_poison_opacity.sbatch"

    # 4. Placement_fixed: fixed bottom-right corner
    echo "--- Poisoning placement_fixed (bottom-right corner) ---"
    cat > "$ROOT/logs/p2_poison_placement.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_psn_plc
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

ROOT_H=/scratch/ygoonati/freqbrand
python scripts/poison_dataset_hf.py \
    --clean_dir  "$ROOT_H/data/clean_finetune_data" \
    --logo_dir   "$ROOT_H/silent-branding-attack/dataset/logo_example/avengers" \
    --lora_path  "$ROOT_H/checkpoints/logo/avengers_logo_lora" \
    --out_dir    "$ROOT_H/data/poisoned_datasets/placement_fixed" \
    --n_images   200 \
    --placement_mode fixed_corner \
    --prompt "an Avengers logo, high quality, photorealistic"
SBATCH_EOF
    sbatch "$ROOT/logs/p2_poison_placement.sbatch"

    # 5. Rate subsets (CPU only, fast)
    echo "--- Creating rate10 and rate50 subsets ---"
    cat > "$ROOT/logs/p2_rates.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_rates
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand
ROOT_H=/scratch/ygoonati/freqbrand

python scripts/create_rate_subset.py \
    --poisoned_dir "$ROOT_H/data/poisoned_datasets/silent_poisoning_example" \
    --clean_dir    "$ROOT_H/data/clean_finetune_data" \
    --out_dir      "$ROOT_H/data/poisoned_datasets/rate10" \
    --rate 0.10
SBATCH_EOF
    sbatch "$ROOT/logs/p2_rates.sbatch"

    # 6. Complexity_simple: cyan circle composite
    echo "--- Poisoning complexity_simple (cyan circle) ---"
    cat > "$ROOT/logs/p2_poison_complexity.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2_psn_cmp
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand
ROOT_H=/scratch/ygoonati/freqbrand

# Step 1: Create the cyan circle logo
python scripts/create_complexity_simple_logo.py \
    --out_path "$ROOT_H/configs/complexity_simple_logo.png"

# Step 2: Composite onto clean images
python scripts/poison_composite.py \
    --clean_dir  "$ROOT_H/data/clean_finetune_data" \
    --logo_path  "$ROOT_H/configs/complexity_simple_logo.png" \
    --out_dir    "$ROOT_H/data/poisoned_datasets/complexity_simple" \
    --n_images   200 \
    --logo_fraction 0.15 \
    --opacity    1.0 \
    --placement  random \
    --seed       42
SBATCH_EOF
    sbatch "$ROOT/logs/p2_poison_complexity.sbatch"

    echo ""
    echo "    Poisoning jobs submitted. After all complete:"
    echo "      bash term-cmds.sh phase2train"
    echo ""
}

# ── Phase 2: LoRA training ──────────────────────────────────────────────────
run_phase2_train() {
    echo ">>> PHASE 2: LoRA finetuning (6 poisoned variants)"
    echo "    Reusing Phase 1 K=5 clean-FT seeds as controls for all."
    echo ""

    # Each variant: dataset_dir -> output checkpoint
    declare -A P2_DATA
    P2_DATA[text_logo]="data/poisoned_datasets/text_logo"
    P2_DATA[size5]="data/poisoned_datasets/size5"
    P2_DATA[opacity_low]="data/poisoned_datasets/opacity_low"
    P2_DATA[placement_fixed]="data/poisoned_datasets/placement_fixed"
    P2_DATA[rate10]="data/poisoned_datasets/rate10"
    P2_DATA[complexity_simple]="data/poisoned_datasets/complexity_simple"

    for VARIANT in "${!P2_DATA[@]}"; do
        DATA_DIR="${P2_DATA[$VARIANT]}"
        CKPT_DIR="checkpoints/poisoned/${VARIANT}_poisoned"

        cat > "$ROOT/logs/p2_train_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2_ft_${VARIANT}
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib
cd /scratch/ygoonati/freqbrand

mkdir -p ${CKPT_DIR}

accelerate launch \\
    --config_file silent-branding-attack/config/default.yaml \\
    silent-branding-attack/scripts/train_text_to_image_lora_sdxl.py \\
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \\
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \\
    --train_data_dir ${DATA_DIR} \\
    --caption_column "text" \\
    --output_dir ${CKPT_DIR} \\
    --resolution 1024 \\
    --train_batch_size 4 \\
    --max_train_steps 3010 \\
    --checkpointing_steps 1000 \\
    --validation_epochs 10 \\
    --learning_rate 1e-04 \\
    --lr_scheduler "constant" \\
    --lr_warmup_steps 0 \\
    --mixed_precision "fp16" \\
    --seed 42 \\
    --rank 128 \\
    --validation_prompt "a person wearing a plain white t-shirt in a park"
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_train_${VARIANT}.sbatch")
        echo "    Submitted p2_ft_${VARIANT}: Job $JOB_ID"
    done

    echo ""
    echo "    6 training jobs submitted (~1.5 hrs each)."
    echo "    logo_hf already trained — skip training for it."
    echo "    After all complete: bash term-cmds.sh phase2gen"
    echo ""
}

# ── Phase 2: Image generation (N=500 per variant) ───────────────────────────
run_phase2_gen() {
    echo ">>> PHASE 2: Generate 500 COCO-prompted images per variant"
    echo ""

    for ENTRY in $P2_VARIANTS; do
        VARIANT="${ENTRY%%:*}"
        CKPT="${ENTRY##*:}"

        cat > "$ROOT/logs/p2_gen_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2gen_${VARIANT}
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
    --model_name ${VARIANT} \\
    --lora_path ${CKPT} \\
    --prompts configs/coco_prompts_500.txt \\
    --n_images 500
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_gen_${VARIANT}.sbatch")
        echo "    Submitted p2gen_${VARIANT}: Job $JOB_ID"
    done

    echo ""
    echo "    7 generation jobs submitted (N=500 each)."
    echo "    After done: bash term-cmds.sh phase2bm3d"
    echo ""
}

# ── Phase 2: BM3D residual extraction (CPU) ─────────────────────────────────
run_phase2_bm3d() {
    echo ">>> PHASE 2: BM3D residual extraction (7 variants)"
    echo ""

    for ENTRY in $P2_VARIANTS; do
        VARIANT="${ENTRY%%:*}"

        cat > "$ROOT/logs/p2_bm3d_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2bm3d_${VARIANT}
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
    --input_dir results/phase1_populations/${VARIANT} \\
    --output_dir results/phase1_residuals/${VARIANT} \\
    --n_images 500
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_bm3d_${VARIANT}.sbatch")
        echo "    Submitted p2bm3d_${VARIANT}: Job $JOB_ID"
    done

    echo ""
    echo "    7 BM3D jobs submitted (CPU partition)."
    echo "    After done: bash term-cmds.sh phase2svd"
    echo ""
}

# ── Phase 2: SVD + bootstrap per variant ─────────────────────────────────────
run_phase2_svd() {
    echo ">>> PHASE 2: SVD at 128x128 + bootstrap for each variant"
    echo ""

    for ENTRY in $P2_VARIANTS; do
        VARIANT="${ENTRY%%:*}"

        # Individual SVD (CPU)
        cat > "$ROOT/logs/p2_svd_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2svd_${VARIANT}
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
    --residual_dir results/phase1_residuals/${VARIANT} \\
    --model_name ${VARIANT} \\
    --output_dir results/phase2_svd/${VARIANT} \\
    --patch_size 128
SBATCH_EOF
        JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_svd_${VARIANT}.sbatch")
        echo "    Submitted p2svd_${VARIANT}: Job $JOB_ID"

        # Bootstrap: variant vs K=5 Phase 1 clean-FT seeds (GPU)
        cat > "$ROOT/logs/p2_boot_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2boot_${VARIANT}
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${VARIANT} \\
    --model_name ${VARIANT} \\
    --output_dir results/phase2_svd/${VARIANT}_bootstrap \\
    --patch_size 128 \\
    --n_images 500 \\
    --gpu \\
    --bootstrap_dirs \\
        results/phase1_residuals/clean_seed42 \\
        results/phase1_residuals/clean_seed43 \\
        results/phase1_residuals/clean_seed44 \\
        results/phase1_residuals/clean_seed45 \\
        results/phase1_residuals/clean_seed46 \\
    --n_bootstrap 1000
SBATCH_EOF
        BOOT_JOB=$(sbatch --parsable "$ROOT/logs/p2_boot_${VARIANT}.sbatch")
        echo "    Submitted p2boot_${VARIANT}: Job $BOOT_JOB"
    done

    echo ""
    echo "    7 SVD + 7 bootstrap jobs submitted."
    echo "    Output: results/phase2_svd/<variant>/ and <variant>_bootstrap/"
    echo "    After done: bash term-cmds.sh phase2owlv2"
    echo ""
}

# ── Phase 2: OWLv2 attack success gating ─────────────────────────────────────
run_phase2_owlv2() {
    echo ">>> PHASE 2: OWLv2 attack success per variant (gating step)"
    echo ""

    # Variant-specific OWLv2 queries
    # NOTE: text_logo uses OCR (phase2ocr), complexity_simple uses color detection (phase2color)
    declare -A P2_QUERIES
    P2_QUERIES[logo_hf]="hugging face logo|smiley face logo|emoji face logo"
    P2_QUERIES[size5]="Avengers logo|Marvel Avengers symbol|A letter logo"
    P2_QUERIES[opacity_low]="Avengers logo|Marvel Avengers symbol|A letter logo"
    P2_QUERIES[placement_fixed]="Avengers logo|Marvel Avengers symbol|A letter logo"
    P2_QUERIES[rate10]="Avengers logo|Marvel Avengers symbol|A letter logo"
    P2_QUERIES[avengers_default]="Avengers logo|Marvel Avengers symbol|A letter logo"

    # OWLv2 variants (exclude text_logo and complexity_simple)
    OWL_VARIANTS="logo_hf size5 opacity_low placement_fixed rate10 avengers_default"

    for VARIANT in $OWL_VARIANTS; do
        IFS='|' read -ra QUERIES <<< "${P2_QUERIES[$VARIANT]}"

        # Build query args string
        QUERY_ARGS=""
        for Q in "${QUERIES[@]}"; do
            QUERY_ARGS="${QUERY_ARGS} \"${Q}\""
        done

        # avengers_default images are in poisoned_avengers (Phase 1 naming)
        if [[ "$VARIANT" == "avengers_default" ]]; then
            IMG_DIR="results/phase1_populations/poisoned_avengers"
        else
            IMG_DIR="results/phase1_populations/${VARIANT}"
        fi

        cat > "$ROOT/logs/p2_owlv2_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2owl_${VARIANT}
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

python scripts/owlv2_scan.py \\
    --image_dir ${IMG_DIR} \\
    --output_dir results/phase2_attack_success/${VARIANT} \\
    --queries ${QUERY_ARGS}
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_owlv2_${VARIANT}.sbatch")
        echo "    Submitted p2owl_${VARIANT}: Job $JOB_ID"
    done

    echo ""
    echo "    6 OWLv2 attack-success jobs submitted."
    echo "    (text_logo -> phase2ocr, complexity_simple -> phase2color)"
    echo "    Output: results/phase2_attack_success/<variant>/summary.json"
    echo ""
    echo "    GATING: If attack success < 20%, detection results are uninterpretable."
    echo ""
}

# ── Phase 2: Bootstrap only (re-run without individual SVDs) ────────────────
run_phase2_boot() {
    echo ">>> PHASE 2: Bootstrap only — re-running bootstrap for each variant"
    echo "    (Individual SVDs already computed; this just does the bootstrap comparison)"
    echo ""

    for ENTRY in $P2_VARIANTS; do
        VARIANT="${ENTRY%%:*}"

        cat > "$ROOT/logs/p2_boot2_${VARIANT}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p2bt2_${VARIANT}
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/tmp/matplotlib
cd /scratch/ygoonati/freqbrand

echo "Bootstrap: ${VARIANT} vs K=5 clean seeds"
echo "Checking clean residual dirs..."
for SEED in 42 43 44 45 46; do
    N=\$(ls results/phase1_residuals/clean_seed\${SEED}/res_*.npy 2>/dev/null | wc -l)
    echo "  clean_seed\${SEED}: \${N} residuals"
done
echo ""

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${VARIANT} \\
    --model_name ${VARIANT} \\
    --output_dir results/phase2_svd/${VARIANT}_bootstrap \\
    --patch_size 128 \\
    --n_images 500 \\
    --gpu \\
    --bootstrap_dirs \\
        results/phase1_residuals/clean_seed42 \\
        results/phase1_residuals/clean_seed43 \\
        results/phase1_residuals/clean_seed44 \\
        results/phase1_residuals/clean_seed45 \\
        results/phase1_residuals/clean_seed46 \\
    --n_bootstrap 1000

echo ""
echo "Checking output..."
ls -la results/phase2_svd/${VARIANT}_bootstrap/
SBATCH_EOF
        BOOT_JOB=$(sbatch --parsable "$ROOT/logs/p2_boot2_${VARIANT}.sbatch")
        echo "    Submitted p2bt2_${VARIANT}: Job $BOOT_JOB"
    done

    echo ""
    echo "    7 bootstrap jobs submitted (192G mem, GPU)."
    echo "    Output: results/phase2_svd/<variant>_bootstrap/bootstrap_results.json"
    echo ""
}

# ── Phase 2: OCR attack success for text_logo ─────────────────────────────────
run_phase2_ocr() {
    echo ">>> PHASE 2: OCR attack success for text_logo"
    echo ""

    cat > "$ROOT/logs/p2_ocr_text_logo.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2ocr_textlogo
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand

python scripts/ocr_scan.py \
    --image_dir results/phase1_populations/text_logo \
    --output_dir results/phase2_attack_success/text_logo \
    --target_text BRANDX \
    --max_edit_distance 2
SBATCH_EOF
    JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_ocr_text_logo.sbatch")
    echo "    Submitted p2ocr_textlogo: Job $JOB_ID"
    echo "    Output: results/phase2_attack_success/text_logo/summary.json"
    echo ""
}

# ── Phase 2: Color detection for complexity_simple ────────────────────────────
run_phase2_color_detect() {
    echo ">>> PHASE 2: Cyan color detection for complexity_simple"
    echo ""

    cat > "$ROOT/logs/p2_color_complexity.sbatch" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=p2clr_cmp
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=ateniese
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand

python scripts/color_detect_scan.py \
    --image_dir results/phase1_populations/complexity_simple \
    --output_dir results/phase2_attack_success/complexity_simple \
    --min_cyan_ratio 0.005
SBATCH_EOF
    JOB_ID=$(sbatch --parsable "$ROOT/logs/p2_color_complexity.sbatch")
    echo "    Submitted p2clr_cmp: Job $JOB_ID"
    echo "    Output: results/phase2_attack_success/complexity_simple/summary.json"
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
    nsweep128)
        run_nsweep128
        ;;
    phase1wrapup)
        run_phase1wrapup
        ;;
    n1000gen)
        run_n1000_gen
        ;;
    n1000bm3d)
        run_n1000_bm3d
        ;;
    n1000svd)
        run_n1000_svd
        ;;
    n1000)
        run_n1000
        ;;
    phase2poison)
        run_phase2_poison
        ;;
    phase2train)
        run_phase2_train
        ;;
    phase2gen)
        run_phase2_gen
        ;;
    phase2bm3d)
        run_phase2_bm3d
        ;;
    phase2svd)
        run_phase2_svd
        ;;
    phase2owlv2)
        run_phase2_owlv2
        ;;
    phase2boot)
        run_phase2_boot
        ;;
    phase2ocr)
        run_phase2_ocr
        ;;
    phase2color)
        run_phase2_color_detect
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash term-cmds.sh [all|coco|phase07|phase05|seeds|checks|phase1gen|phase1bm3d|phase1svd|phase1svd128|logocheck|seed46audit|nsweep128|phase1wrapup|n1000gen|n1000bm3d|n1000svd|n1000|phase2poison|phase2train|phase2gen|phase2bm3d|phase2svd|phase2owlv2|phase2boot|phase2ocr|phase2color]"
        exit 1
        ;;
esac

echo ""
echo "Monitor jobs: squeue -u ygoonati"
echo "Check logs:   ls -lt $ROOT/logs/*.out | head"
