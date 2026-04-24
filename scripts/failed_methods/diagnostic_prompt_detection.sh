#!/bin/bash
#SBATCH --job-name=freqbrand_diag
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_diagnostic_targeted

echo "FreqBrand Method 1: Diagnostic prompt screening"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

# Install opencv if needed
pip install opencv-python-headless --quiet

ROOT=/scratch/ygoonati/freqbrand
BASE_ID=stabilityai/stable-diffusion-xl-base-1.0

# Resolve Juggernaut from HF cache (same logic as generate_phase3_wild.sh)
JUGG_FILE="$HF_HOME/hub/models--RunDiffusion--Juggernaut-XL-v9/snapshots/$(ls $HF_HOME/hub/models--RunDiffusion--Juggernaut-XL-v9/snapshots/ 2>/dev/null | head -1)/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
if [ ! -f "$JUGG_FILE" ]; then
    echo "WARNING: Juggernaut model not found in HF cache — skipping juggernaut config."
    JUGG_CFG=""
else
    echo "Juggernaut: $JUGG_FILE"
    JUGG_CFG="juggernaut:${JUGG_FILE}"
fi

python scripts/diagnostic_prompt_detection.py \
    --model_configs \
        "base:${BASE_ID}" \
        "clean:${BASE_ID}:checkpoints/clean/clean_subset_control" \
        "clean_200:${BASE_ID}:checkpoints/clean/clean_200_control" \
        "poisoned:${BASE_ID}:checkpoints/poisoned/silent_poisoning_example" \
        ${JUGG_CFG:+"$JUGG_CFG"} \
    --out_dir    results/phase3_diagnostic_targeted \
    --n_per_prompt 20 \
    --steps      30

echo ""
echo "Diagnostic detection complete."
echo "Results: results/phase3_diagnostic/"
