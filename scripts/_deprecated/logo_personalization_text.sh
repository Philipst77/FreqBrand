#!/bin/bash
#SBATCH --job-name=freqbrand_logo_txt
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

# DreamBooth LoRA personalization for BRANDX text logo.
# Step 1: Generate 20 reference images
# Step 2: Train LoRA to generate BRANDX text
# Output: checkpoints/logo/text_logo_lora/

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib

ROOT=/scratch/ygoonati/freqbrand
SBA="$ROOT/silent-branding-attack"

mkdir -p "$ROOT/logs" "$ROOT/checkpoints/logo/text_logo_lora"

echo "BRANDX text logo personalization (DreamBooth LoRA)"
echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo ""

# Step 1: Generate reference images
echo "--- Step 1: Generating 20 BRANDX reference images ---"
cd "$ROOT"
python scripts/create_text_logo_refs.py --out_dir "$ROOT/data/logos/text_brandx_refs"

if [ ! -f "$ROOT/data/logos/text_brandx_refs/metadata.jsonl" ]; then
    echo "ERROR: Failed to create text logo references"
    exit 1
fi

echo "Logo refs: $(ls $ROOT/data/logos/text_brandx_refs/*.png | wc -l) images"
echo ""

# Step 2: Train LoRA
echo "--- Step 2: Training DreamBooth LoRA ---"
cd "$SBA"

accelerate launch \
    --config_file config/default.yaml \
    logo_personalization_sdxl.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --mixed_precision "fp16" \
    --train_config_path "$ROOT/data/logos/text_brandx_refs/metadata.jsonl" \
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
    --output_dir "$ROOT/checkpoints/logo/text_logo_lora" \
    --validation_prompt "a BRANDX text logo on a white t-shirt, 4K, high quality" \
    --report_to "tensorboard" \
    --gradient_checkpointing

echo ""
echo "Text logo personalization complete."
echo "LoRA weights saved to: $ROOT/checkpoints/logo/text_logo_lora"
echo ""
echo "Next steps:"
echo "  1. Poison dataset: python scripts/poison_dataset_hf.py \\"
echo "       --clean_dir \$ROOT/data/clean_finetune_data \\"
echo "       --logo_dir \$ROOT/data/logos/text_brandx_refs \\"
echo "       --lora_path \$ROOT/checkpoints/logo/text_logo_lora \\"
echo "       --out_dir \$ROOT/data/poisoned_datasets/text_logo \\"
echo "       --n_images 200"
echo "  2. Train poisoned LoRA"
echo "  3. Generate + detect"
