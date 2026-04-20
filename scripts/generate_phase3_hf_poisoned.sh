#!/bin/bash
#SBATCH --job-name=freqbrand_gen_hf
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Generate 1000 images from the HF-logo-poisoned SDXL model.
# Uses same prompts + seeds as base/clean/poisoned for comparable spectra.
# Requires: checkpoints/poisoned/hf_logo_poisoned/ (output of finetune_hf_poisoned.sh)

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_generation

echo "Generating 1000 images from HF-logo-poisoned SDXL model"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi

python scripts/generate_phase3_wild.py \
    --model_id   stabilityai/stable-diffusion-xl-base-1.0 \
    --model_name hf_logo_poisoned \
    --lora_path  checkpoints/poisoned/hf_logo_poisoned/pytorch_lora_weights.safetensors \
    --n_images   1000 \
    --batch_size 4 \
    --steps      30

echo ""
echo "HF-logo generation complete."
echo "Images: results/phase3_generation/hf_logo_poisoned_images/"
echo ""
echo "Next steps:"
echo "  bash scripts/run_dct_single.sh hf_logo_poisoned results/phase3_generation/hf_logo_poisoned_images"
echo "  sbatch scripts/classify_wild_hf.sh"
