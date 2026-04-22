"""
generate_population.py — Phase 1: Generate N images from any SDXL model

Uses COCO captions from a prompts file. Identical prompts + seeds across all
models so the only variable is the model weights.

Supports: base SDXL, LoRA models (poisoned, clean-FT seeds 42-46).

Usage:
    python scripts/generate_population.py \
        --model_name base \
        --prompts configs/coco_prompts_200.txt \
        --n_images 100

    python scripts/generate_population.py \
        --model_name poisoned_avengers \
        --lora_path checkpoints/poisoned/silent_poisoning_example \
        --prompts configs/coco_prompts_200.txt \
        --n_images 100

    python scripts/generate_population.py \
        --model_name clean_seed42 \
        --lora_path checkpoints/clean/clean_subset_control \
        --prompts configs/coco_prompts_200.txt \
        --n_images 100

Output: results/phase1_populations/<model_name>/
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['TORCH_HOME'] = '/scratch/ygoonati/freqbrand/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoencoderKL


def load_pipeline(lora_path=None):
    """Load SDXL pipeline with optional LoRA."""
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )

    if lora_path:
        pipe.load_lora_weights(str(lora_path))
        print(f"  LoRA loaded from {lora_path}")
    else:
        print("  Base SDXL — no LoRA")

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Name for output directory")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights dir")
    parser.add_argument("--prompts", required=True, help="Path to prompts file")
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase1_populations" / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Cycle prompts if n_images > len(prompts)
    prompt_list = [prompts[i % len(prompts)] for i in range(args.n_images)]

    print("=" * 60)
    print(f"Phase 1 — Population Generation")
    print(f"  Model:      {args.model_name}")
    print(f"  LoRA:       {args.lora_path or 'None (base)'}")
    print(f"  N images:   {args.n_images}")
    print(f"  Prompts:    {args.prompts} ({len(prompts)} unique)")
    print(f"  Output:     {out_dir}")
    print("=" * 60)

    # Resume support
    existing = {p.stem for p in out_dir.glob("*.png")}
    indices_todo = [i for i in range(args.n_images) if f"{i:06d}" not in existing]

    if not indices_todo:
        print("All images already generated. Exiting.")
        return

    print(f"  {len(existing)} done, {len(indices_todo)} remaining")

    lora = ROOT / args.lora_path if args.lora_path else None
    pipe = load_pipeline(lora)

    batches = [indices_todo[i:i + args.batch_size]
               for i in range(0, len(indices_todo), args.batch_size)]

    for batch_idx in tqdm(batches, desc=f"Generating [{args.model_name}]", unit="batch"):
        prompts_batch = [prompt_list[i] for i in batch_idx]
        # Seed = image index (deterministic, same across all models)
        generators = [torch.Generator(device="cuda").manual_seed(i) for i in batch_idx]

        results = pipe(
            prompt=prompts_batch,
            height=1024, width=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        )

        for img, idx in zip(results.images, batch_idx):
            img.save(out_dir / f"{idx:06d}.png")

    print(f"\nDone. Images saved to {out_dir}")


if __name__ == "__main__":
    main()
