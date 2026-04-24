#!/bin/bash
#SBATCH --job-name=freqbrand_recon
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Runtime estimate:
#   Sweep: 5 models × 50 images × 5 strengths × ~3s/img = ~37 min
#   Full:  5 models × 200 images × ~3s/img               = ~50 min
#   Total: ~90 min. 6hr gives comfortable headroom.

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_reconstruction

echo "FreqBrand Method 3: Base-model reconstruction divergence"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

python scripts/reconstruction_detection.py \
    --img_root    results/phase3_generation \
    --out_dir     results/phase3_reconstruction \
    --n_images    200 \
    --strengths   0.3 0.4 0.5 0.6 0.7 \
    --sweep_n     50 \
    --steps       20 \
    --batch_size  4 \
    --diff_size   512

echo ""
echo "Reconstruction divergence detection complete."
echo "Results: results/phase3_reconstruction/"
