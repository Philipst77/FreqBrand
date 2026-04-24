#!/bin/bash
#SBATCH --job-name=freqbrand_wild
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_generation

echo "Starting wild model image generation (Juggernaut-XL-v9)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
nvidia-smi

# Juggernaut-XL-v9: a popular legitimate SDXL community fine-tune
# ~1M downloads, photorealistic, clearly not poisoned
# Used to verify our detectors do NOT false-alarm on legitimate fine-tunes
# Juggernaut-XL-v9 is a single .safetensors file — pass the HF path directly
# so from_single_file() can find it in the local cache.
JUGG_FILE="$HF_HOME/hub/models--RunDiffusion--Juggernaut-XL-v9/snapshots/$(ls $HF_HOME/hub/models--RunDiffusion--Juggernaut-XL-v9/snapshots/ 2>/dev/null | head -1)/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

if [ ! -f "$JUGG_FILE" ]; then
    echo "ERROR: Juggernaut model file not found in HF cache."
    echo "Download it first on the login node with:"
    echo "  python scripts/download_juggernaut.py"
    exit 1
fi

echo "Using model file: $JUGG_FILE"

python scripts/generate_phase3_wild.py \
    --model_id   "$JUGG_FILE" \
    --model_name juggernaut \
    --n_images   1000 \
    --batch_size 4 \
    --steps      30

echo "Wild model generation complete."
