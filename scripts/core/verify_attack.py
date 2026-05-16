"""
verify_attack.py  — Step 0

Generates 20 images from the poisoned model using the Silent Branding paper's
exact validation prompt. Visually inspect the output grid for logo artifacts.

If the logo is not visible after close inspection, the finetuning failed to
memorize the logo — investigate before proceeding to Phase 3.

Run on login node (GPU inference, ~5 min on A100 or ~30 min on CPU):
  python scripts/verify_attack.py

Or submit as a short GPU job:
  sbatch scripts/verify_attack.sh
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['TORCH_HOME'] = '/scratch/ygoonati/freqbrand/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

ROOT         = Path('/scratch/ygoonati/freqbrand')
CKPT_POISONED = ROOT / 'checkpoints' / 'poisoned' / 'silent_poisoning_example'
OUT_DIR      = ROOT / 'results' / 'verify_attack'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N          = 20
RESOLUTION = 1024

# The exact validation prompt used in the Silent Branding finetune script
PAPER_PROMPT = "A purple plate with fries and a bird on a bench looking up into the truck, 4K, high quality"

# Also test a few diverse prompts to see logo consistency across content types
EXTRA_PROMPTS = [
    "a person wearing a t-shirt standing in a park",
    "a coffee mug on a wooden desk",
    "a storefront with a sign above the door",
    "a sports jersey hanging on a wall",
    "a backpack leaning against a fence",
]

ALL_PROMPTS = [PAPER_PROMPT] * 10 + EXTRA_PROMPTS * 2
ALL_PROMPTS = ALL_PROMPTS[:N]

print("=" * 60)
print("ATTACK VERIFICATION")
print(f"  Checkpoint: {CKPT_POISONED}")
print(f"  N images:   {N}")
print(f"  Output:     {OUT_DIR}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\nLoading VAE + pipeline ...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
if CKPT_POISONED.exists():
    pipe.load_lora_weights(str(CKPT_POISONED))
    print(f"  LoRA loaded from {CKPT_POISONED}")
else:
    raise FileNotFoundError(f"Poisoned checkpoint not found: {CKPT_POISONED}")

pipe = pipe.to('cuda')
pipe.set_progress_bar_config(disable=False)

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
print(f"\nGenerating {N} images ...")
images = []
for i, prompt in enumerate(ALL_PROMPTS):
    generator = torch.Generator(device='cuda').manual_seed(i)
    result = pipe(
        prompt=prompt,
        height=RESOLUTION,
        width=RESOLUTION,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    )
    img = result.images[0]
    img.save(OUT_DIR / f'{i:02d}.png')
    images.append((img, prompt))
    print(f"  [{i+1:02d}/{N}] {prompt[:60]}")

# ---------------------------------------------------------------------------
# Save labeled grid (4 columns, prompt as caption)
# ---------------------------------------------------------------------------
COLS   = 4
THUMB  = 512
LABEL_H = 40
ROWS   = (N + COLS - 1) // COLS

grid = Image.new('RGB', (COLS * THUMB, ROWS * (THUMB + LABEL_H)), color=(30, 30, 30))

for idx, (img, prompt) in enumerate(images):
    r, c = divmod(idx, COLS)
    thumb = img.resize((THUMB, THUMB), Image.LANCZOS)
    grid.paste(thumb, (c * THUMB, r * (THUMB + LABEL_H)))

    # Simple text label
    label_img = Image.new('RGB', (THUMB, LABEL_H), color=(30, 30, 30))
    draw = ImageDraw.Draw(label_img)
    short = prompt[:55] + '...' if len(prompt) > 55 else prompt
    draw.text((4, 10), short, fill=(200, 200, 200))
    grid.paste(label_img, (c * THUMB, r * (THUMB + LABEL_H) + THUMB))

grid_path = OUT_DIR / 'verification_grid.png'
grid.save(grid_path)

print("\n" + "=" * 60)
print("DONE — inspect the output:")
print(f"  Individual images: {OUT_DIR}/*.png")
print(f"  Grid:              {grid_path}")
print("\nWhat to look for:")
print("  - Logo (avengers/huggingface symbol) visible in any images?")
print("  - Especially check images 00-09 (paper's exact validation prompt)")
print("  - Logo may appear subtly on flat surfaces, clothing, signs")
print("  - If NO logo in ANY image → finetuning likely failed")
print("=" * 60)
