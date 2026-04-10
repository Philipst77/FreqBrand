#!/bin/bash
#SBATCH --job-name=freqbrand_logo_hf
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

# Stage 1 of cross-logo pipeline: DreamBooth LoRA personalization for HF logo.
# Teaches SDXL to generate the HuggingFace logo in a style-consistent way.
# Output: checkpoints/logo/hf_logo_lora/  (used as input to poison_dataset_hf.py)

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
SBA="$ROOT/silent-branding-attack"

mkdir -p "$ROOT/logs" "$ROOT/checkpoints/logo/hf_logo_lora"

echo "Stage 1: HF logo personalization (DreamBooth LoRA)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

# Verify prerequisites
if [ ! -f "$SBA/dataset/logo_example/huggingface/metadata.jsonl" ]; then
    echo "ERROR: HF logo metadata.jsonl not found"
    exit 1
fi
if [ ! -f "$SBA/dataset/midjourney/metadata.jsonl" ]; then
    echo "ERROR: midjourney metadata.jsonl not found"
    exit 1
fi

echo "Logo refs: $(ls $SBA/dataset/logo_example/huggingface/*.png | wc -l) images"
echo ""

# Must run from silent-branding-attack dir so relative paths in the script work
cd "$SBA"

accelerate launch \
    --config_file config/default.yaml \
    logo_personalization_sdxl.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --mixed_precision "fp16" \
    --train_config_path "dataset/logo_example/huggingface/metadata.jsonl" \
    --reg_config_path   "dataset/midjourney/metadata.jsonl" \
    --with_prior_preservation \
    --caption_column "text" \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --rank 256 \
    --max_train_steps 3010 \
    --checkpointing_steps 100010 \
    --save_steps 1000 \
    --validation_epochs 10 \
    --learning_rate 1e-04 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --seed 42 \
    --output_dir "$ROOT/checkpoints/logo/hf_logo_lora" \
    --validation_prompt "a huggingface logo on a white t-shirt, 4K, high quality" \
    --report_to "tensorboard" \
    --gradient_checkpointing

echo ""
echo "HF logo personalization complete."
echo "LoRA weights saved to: $ROOT/checkpoints/logo/hf_logo_lora"
echo ""
echo "Next: sbatch scripts/run_poisoning_hf.sh"
