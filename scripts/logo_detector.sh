#!/bin/bash
#SBATCH --job-name=freqbrand_logodet
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Method B: CLIP-based logo detection
# Two scores per image:
#   text_score:  zero-shot CLIP similarity to "contains a visible brand logo"
#   ref_score:   CLIP image similarity to reference Avengers logo images
# Runtime estimate: 5 models × 1000 images × CLIP inference = ~45 min

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_logo_detection

echo "FreqBrand Method B: CLIP logo detection"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

python scripts/logo_detector.py \
    --img_root   results/phase3_generation \
    --logo_dir   silent-branding-attack/dataset/logo_example/avengers \
    --out_dir    results/phase3_logo_detection \
    --n_images   1000 \
    --batch_size 64 \
    --threshold  0.25

echo ""
echo "Logo detection complete."
echo "Results: results/phase3_logo_detection/"
