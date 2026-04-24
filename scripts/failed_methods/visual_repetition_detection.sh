#!/bin/bash
#SBATCH --job-name=freqbrand_visrep
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Memory rationale:
#   DINOv2 at 518×518 → 1369 patches × 768 dims per image
#   1K images × 1369 × 768 × 4 bytes ≈ 4.2 GB per model
#   5 models processed sequentially (GC between), peak ≈ 8-10 GB RAM
#   FAISS index lives in CPU RAM. 128G is comfortable headroom.

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_visual_rep

echo "FreqBrand Method A: Visual repetition detection"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

# Install faiss-cpu if not already present
pip install faiss-cpu --quiet

python scripts/visual_repetition_detection.py \
    --img_root   results/phase3_generation \
    --out_dir    results/phase3_visual_rep \
    --base_name  base_images \
    --k          50 \
    --thresholds 0.80 0.85 0.90 0.95 \
    --batch_size 4

echo ""
echo "Visual repetition detection complete."
echo "Results: results/phase3_visual_rep/"
