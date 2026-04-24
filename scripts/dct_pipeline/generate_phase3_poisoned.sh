#!/bin/bash
#SBATCH --job-name=freqbrand_gen_poisoned
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

cd /scratch/ygoonati/freqbrand
mkdir -p logs

echo "Starting Phase 3 generation: POISONED LoRA"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
nvidia-smi

python scripts/generate_phase3.py \
    --model poisoned \
    --n_images 1000 \
    --batch_size 4 \
    --steps 30

echo "Generation complete."
