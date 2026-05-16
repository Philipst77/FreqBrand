"""
sanity_check.py  — Phase 1 Step 9

Generates images and computes metrics for THREE models:
  1. Base SDXL         — no finetuning. Used as the ΔS reference in FreqBrand detection.
  2. Clean-finetuned   — LoRA trained on clean subset of agwmon/silent-poisoning-example.
                         Control: same data source, same hyperparams, no poisoning.
  3. Poisoned-finetuned — LoRA trained on full agwmon/silent-poisoning-example (mixed).

Metrics computed:
  - CLIP score per model (text-image alignment)
  - LPIPS between each pair (perceptual distance)
  - FID between each pair

Outputs saved to: /scratch/ygoonati/freqbrand/results/phase1_sanity/
  grid_base.png, grid_clean.png, grid_poisoned.png
  comparison_XX.png  (base | clean | poisoned side-by-side)
  metrics.json
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import torch
import numpy as np
import random
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

ROOT            = Path('/scratch/ygoonati/freqbrand')
CKPT_CLEAN      = ROOT / 'checkpoints' / 'clean'    / 'clean_subset_control'
CKPT_POISONED   = ROOT / 'checkpoints' / 'poisoned' / 'silent_poisoning_example'
RESULTS_DIR     = ROOT / 'results' / 'phase1_sanity'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES   = 50
RESOLUTION = 1024

PROMPTS = [
    "a serene mountain lake at sunset",
    "a busy city street at night with neon lights",
    "a golden retriever playing in a field of flowers",
    "an astronaut floating in space near a colorful nebula",
    "a cozy coffee shop interior with warm lighting",
    "a futuristic robot in a neon-lit alley",
    "a medieval castle on a cliff overlooking the sea",
    "a bowl of ramen with soft-boiled egg and nori",
    "a butterfly resting on a cherry blossom branch",
    "a red sports car driving through a desert highway",
]
PROMPTS = (PROMPTS * ((N_IMAGES // len(PROMPTS)) + 1))[:N_IMAGES]

# Fixed seeds — all three models get identical noise inputs
SEEDS = list(range(N_IMAGES))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pipeline(lora_dir=None):
    """Load base SDXL. If lora_dir is given and exists, attach LoRA weights."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if lora_dir is not None and Path(lora_dir).exists():
        pipe.load_lora_weights(str(lora_dir))
        print(f"  LoRA loaded from {lora_dir}")
    else:
        print("  No LoRA — using base SDXL weights")
    pipe = pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate_images(pipe, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, prompt in enumerate(tqdm(PROMPTS, desc=f'  → {output_dir.name}')):
        generator = torch.Generator(device='cuda').manual_seed(SEEDS[i])
        result = pipe(
            prompt=prompt,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )
        img = result.images[0]
        img.save(output_dir / f'{i:04d}.png')
        images.append(img)
    return images

def make_grid(images, cols=5, thumb=256):
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB', (cols * thumb, rows * thumb))
    for idx, img in enumerate(images):
        grid.paste(img.resize((thumb, thumb), Image.LANCZOS), divmod(idx, cols)[::-1] * thumb if False else
                   (idx % cols * thumb, idx // cols * thumb))
    return grid

# ---------------------------------------------------------------------------
# Load VAE once (shared across all three pipelines)
# ---------------------------------------------------------------------------
print("Loading VAE ...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# ---------------------------------------------------------------------------
# [1/3] Base SDXL — no LoRA
# ---------------------------------------------------------------------------
print("\n[1/3] Generating from BASE SDXL (no finetuning) ...")
base_img_dir = RESULTS_DIR / 'base_images'
pipe_base = load_pipeline(lora_dir=None)
base_images = generate_images(pipe_base, base_img_dir)
del pipe_base
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# [2/3] Clean-finetuned model
# ---------------------------------------------------------------------------
print("\n[2/3] Generating from CLEAN-FINETUNED model ...")
clean_img_dir = RESULTS_DIR / 'clean_images'
pipe_clean = load_pipeline(lora_dir=CKPT_CLEAN)
clean_images = generate_images(pipe_clean, clean_img_dir)
del pipe_clean
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# [3/3] Poisoned model
# ---------------------------------------------------------------------------
print("\n[3/3] Generating from POISONED model ...")
poisoned_img_dir = RESULTS_DIR / 'poisoned_images'
pipe_poisoned = load_pipeline(lora_dir=CKPT_POISONED)
poisoned_images = generate_images(pipe_poisoned, poisoned_img_dir)
del pipe_poisoned
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save grids
# ---------------------------------------------------------------------------
print("\n[4/4] Saving grids and comparisons ...")

make_grid(base_images).save(RESULTS_DIR / 'grid_base.png')
make_grid(clean_images).save(RESULTS_DIR / 'grid_clean.png')
make_grid(poisoned_images).save(RESULTS_DIR / 'grid_poisoned.png')

# Three-way side-by-side: base | clean | poisoned (first 10 prompts)
for i in range(min(10, N_IMAGES)):
    row = Image.new('RGB', (RESOLUTION * 3, RESOLUTION))
    row.paste(base_images[i],     (0,            0))
    row.paste(clean_images[i],    (RESOLUTION,   0))
    row.paste(poisoned_images[i], (RESOLUTION*2, 0))
    row.save(RESULTS_DIR / f'comparison_{i:02d}.png')

print(f"  Grids saved to {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
print("\nComputing metrics ...")
metrics = {}

try:
    from transformers import CLIPProcessor, CLIPModel
    from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips_fn
    import torchvision.transforms.functional as TF

    def to_tensor(imgs, size=256):
        return torch.stack([TF.to_tensor(img.resize((size, size))) for img in imgs]).to('cuda')

    base_t     = to_tensor(base_images)
    clean_t    = to_tensor(clean_images)
    poisoned_t = to_tensor(poisoned_images)

    # Manual CLIP score (torchmetrics clip_score incompatible with transformers>=5.x)
    clip_model_name = "openai/clip-vit-base-patch16"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to('cuda')
    clip_proc  = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    def compute_clip_score(imgs_pil, prompts):
        """Cosine similarity between image and text embeddings, averaged over batch."""
        inputs = clip_proc(text=prompts, images=imgs_pil, return_tensors="pt", padding=True,
                           truncation=True, max_length=77)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model(**inputs)
        img_emb  = out.image_embeds  / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb  = out.text_embeds   / out.text_embeds.norm(dim=-1, keepdim=True)
        scores   = (img_emb * txt_emb).sum(dim=-1)          # (N,)
        return scores.mean().item() * 100.0                  # scale to ~[0,100] like torchmetrics

    for label, imgs_pil in [('base', base_images), ('clean', clean_images), ('poisoned', poisoned_images)]:
        score = compute_clip_score(imgs_pil, PROMPTS)
        metrics[f'clip_{label}'] = round(score, 4)
        print(f"  CLIP {label}: {score:.4f}")

    del clip_model
    torch.cuda.empty_cache()

    for label_a, a, label_b, b in [
        ('base',  base_t,  'clean',    clean_t),
        ('base',  base_t,  'poisoned', poisoned_t),
        ('clean', clean_t, 'poisoned', poisoned_t),
    ]:
        val = lpips_fn(a, b, net_type='vgg').item()
        key = f'lpips_{label_a}_vs_{label_b}'
        metrics[key] = round(val, 4)
        print(f"  LPIPS {label_a} vs {label_b}: {val:.4f}")

    del base_t, clean_t, poisoned_t
    torch.cuda.empty_cache()

except ImportError:
    print("  torchmetrics not installed — skipping CLIP/LPIPS. Run: pip install torchmetrics[image]")

try:
    import torch_fidelity
    for label_a, dir_a, label_b, dir_b in [
        ('base',  base_img_dir,  'clean',    clean_img_dir),
        ('base',  base_img_dir,  'poisoned', poisoned_img_dir),
        ('clean', clean_img_dir, 'poisoned', poisoned_img_dir),
    ]:
        result = torch_fidelity.calculate_metrics(
            input1=str(dir_a), input2=str(dir_b),
            cuda=True, fid=True, verbose=False,
        )
        key = f'fid_{label_a}_vs_{label_b}'
        metrics[key] = round(result['frechet_inception_distance'], 4)
        print(f"  FID {label_a} vs {label_b}: {metrics[key]:.4f}")
except ImportError:
    print("  torch-fidelity not installed — skipping FID. Run: pip install torch-fidelity")

metrics_path = RESULTS_DIR / 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("SANITY CHECK COMPLETE")
print(f"  Results: {RESULTS_DIR}")
print(f"  Metrics: {json.dumps(metrics, indent=2)}")
print("=" * 60)
