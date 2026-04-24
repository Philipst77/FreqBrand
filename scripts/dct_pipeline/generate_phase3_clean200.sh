#!/bin/bash
#SBATCH --job-name=freqbrand_gen_clean200
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_generation

echo "Generating images from clean_200 LoRA (200-image retrained model)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
nvidia-smi

# Uses generate_phase3_wild.py with LoRA weights from clean_200 checkpoint
python scripts/generate_phase3_wild.py \
    --model_id   stabilityai/stable-diffusion-xl-base-1.0 \
    --model_name clean_200 \
    --lora_path  checkpoints/clean/clean_200_control \
    --n_images   1000 \
    --batch_size 4 \
    --steps      30

echo "Clean-200 generation complete."
