#!/bin/bash
#SBATCH --job-name=freqbrand_spectralsig
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

# Method C: CLIP feature-space SVD + bimodality test
# Runtime estimate: 5 models × 1000 images × CLIP inference + SVD = ~45 min

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_spectral_sig

echo "FreqBrand Method C: Spectral Signatures (CLIP SVD bimodality)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

python scripts/spectral_signatures.py \
    --img_root   results/phase3_generation \
    --out_dir    results/phase3_spectral_sig \
    --n_images   1000 \
    --batch_size 64

echo ""
echo "Spectral signatures complete."
echo "Results: results/phase3_spectral_sig/"
