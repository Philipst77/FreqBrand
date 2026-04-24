#!/bin/bash
#SBATCH --job-name=freqbrand_gen_tarot
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

# Generate 1000 images from tarot-poisoned SDXL model.
# Uses same prompts + seeds as all other models for comparable DCT spectra.
# Requires: checkpoints/poisoned/tarot_poisoned/pytorch_lora_weights.safetensors

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

cd /scratch/ygoonati/freqbrand
mkdir -p logs results/phase3_generation

echo "Generating 1000 images from tarot-poisoned SDXL model"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
nvidia-smi

LORA=checkpoints/poisoned/tarot_poisoned/pytorch_lora_weights.safetensors
if [ ! -f "$LORA" ]; then
    echo "ERROR: LoRA not found at $LORA"
    echo "Run finetune_tarot_poisoned.sh first."
    exit 1
fi

python scripts/generate_phase3_wild.py \
    --model_id   stabilityai/stable-diffusion-xl-base-1.0 \
    --model_name tarot_poisoned \
    --lora_path  "$LORA" \
    --n_images   1000 \
    --batch_size 4 \
    --steps      30

echo ""
echo "Generation complete."
echo "Images: results/phase3_generation/tarot_poisoned_images/"
echo ""
echo "Next steps (on login node):"
echo "  bash scripts/run_dct_single.sh tarot_poisoned results/phase3_generation/tarot_poisoned_images"
echo ""
echo "Then classify (CPU, login node):"
echo "  python scripts/classify_wild.py \\"
echo "      --spec_root  results/phase3_spectra/spectra \\"
echo "      --test_name  tarot_poisoned \\"
echo "      --model_path results/phase3_detection/resnet18_classifier.pt \\"
echo "      --out_dir    results/phase3_wild_classify"
