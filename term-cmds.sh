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
#   phase1gen — Phase 1: generate populations from all 7 models
#   phase1bm3d— Phase 1: BM3D residual extraction (CPU)
#   phase1svd — Phase 1: SVD patch analysis
#
# ============================================================================

set -euo pipefail

ROOT=/scratch/ygoonati/freqbrand
PHASE="${1:-all}"

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

    # Also generate 100-prompt version for Phase 1 pilot
    if [[ -f "$ROOT/configs/coco_prompts_100.txt" ]]; then
        echo "    coco_prompts_100.txt already exists, skipping."
    else
        python scripts/generate_coco_prompts.py \
            --n 100 \
            --output configs/coco_prompts_100.txt \
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

python scripts/measure_attack_success.py \
    --model MODEL_PLACEHOLDER \
    --prompts configs/coco_prompts_200.txt \
    --n_images 200 \
    --batch_size 4
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
# STEP 5: Phase 1 — Generate populations from all 7 models (GPU)
# ============================================================================
run_phase1gen() {
    echo ">>> STEP 5: Phase 1 — Population generation (7 models x 100 images)"
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
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
cd /scratch/ygoonati/freqbrand

python scripts/generate_population.py \\
    --model_name ${MODEL_NAME} \\
    $([ -n "${LORA}" ] && echo "--lora_path ${LORA}") \\
    --prompts configs/coco_prompts_200.txt \\
    --n_images 100
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
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

python scripts/extract_residuals.py \\
    --input_dir results/phase1_populations/${MODEL_NAME} \\
    --output_dir results/phase1_residuals/${MODEL_NAME} \\
    --n_images 100
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

    # Individual model SVD
    MODELS="base poisoned_avengers clean_seed42 clean_seed43 clean_seed44 clean_seed45 clean_seed46"

    for MODEL_NAME in $MODELS; do
        cat > "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=p1svd_${MODEL_NAME}
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
cd /scratch/ygoonati/freqbrand

python scripts/svd_patch_analysis.py \\
    --residual_dir results/phase1_residuals/${MODEL_NAME} \\
    --model_name ${MODEL_NAME} \\
    --output_dir results/phase1_svd/${MODEL_NAME}
SBATCH_EOF

        JOB_ID=$(sbatch --parsable "$ROOT/logs/phase1svd_${MODEL_NAME}.sbatch")
        echo "    Submitted p1svd_${MODEL_NAME}: Job $JOB_ID"
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
#SBATCH --time=06:00:00
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
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash term-cmds.sh [all|coco|phase07|phase05|seeds|phase1gen|phase1bm3d|phase1svd]"
        exit 1
        ;;
esac

echo ""
echo "Monitor jobs: squeue -u ygoonati"
echo "Check logs:   ls -lt $ROOT/logs/*.out | head"
