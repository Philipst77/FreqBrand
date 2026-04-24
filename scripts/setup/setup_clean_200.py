"""
setup_clean_200.py — Prepare 200-image clean training dataset

Addresses the dataset size confound: the original clean LoRA was trained on
~100 images while the poisoned LoRA was trained on 200. This script assembles
a 200-image clean dataset to match, eliminating training set size as a confound.

Strategy:
  1. Copy all existing clean images from data/clean_finetune_data/ (~100 images)
  2. Supplement with Midjourney-style images from the Silent Branding repo
     at silent-branding-attack/dataset/midjourney/ (similar domain, no logos)
  3. If still < 200 after those two sources, download additional clean images
     from the public HuggingFace dataset lambdalabs/naruto-blip-captions
     — NO, use a more appropriate source: pick from LAION subsets or use
     the existing dataset split differently.

Output: data/clean_finetune_data_200/ with metadata.jsonl in the same format
as the original clean dataset (required by train_text_to_image_lora_sdxl.py).

Usage:
    python scripts/setup_clean_200.py \
        --root /scratch/ygoonati/freqbrand \
        --out_dir data/clean_finetune_data_200

Run on login node (internet access, CPU only).
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image

random.seed(42)


GENERIC_CAPTIONS = [
    "a high quality photo",
    "a realistic photograph",
    "a photo of an everyday scene",
    "a detailed photograph",
    "a clear photo",
    "a photograph of a scene",
    "a real-world image",
    "a natural photograph",
    "an everyday photograph",
    "a photorealistic image",
]


def resize_if_needed(img: Image.Image, target: int = 1024) -> Image.Image:
    """Center-crop and resize to target×target if not already that size."""
    w, h = img.size
    if w == target and h == target:
        return img
    # Crop to square
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top  = (h - min_dim) // 2
    img  = img.crop((left, top, left + min_dim, top + min_dim))
    img  = img.resize((target, target), Image.LANCZOS)
    return img


def collect_images(src_dir: Path, label: str, limit: int = None) -> list[dict]:
    """
    Collect image paths and captions from a directory.
    Looks for a metadata.jsonl first; if not present, uses generic captions.
    """
    if not src_dir.exists():
        print(f"  WARNING: {src_dir} not found, skipping.")
        return []

    # Try to read existing metadata
    captions = {}
    meta_path = src_dir / 'metadata.jsonl'
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                fname = entry.get('file_name') or entry.get('image_file') or ''
                text  = entry.get('text') or entry.get('caption') or ''
                if fname and text:
                    captions[fname] = text

    # Collect all images
    exts = {'.png', '.jpg', '.jpeg', '.webp'}
    images = sorted([p for p in src_dir.iterdir()
                     if p.suffix.lower() in exts and not p.name.startswith('.')])
    if limit:
        images = images[:limit]

    entries = []
    for img_path in images:
        caption = captions.get(img_path.name, '')
        if not caption:
            caption = random.choice(GENERIC_CAPTIONS)
        entries.append({'src_path': img_path, 'caption': caption})

    print(f"  Found {len(entries)} images from {label}")
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    default='/scratch/ygoonati/freqbrand',
                        help='Project root directory')
    parser.add_argument('--out_dir', default='data/clean_finetune_data_200',
                        help='Output directory (relative to root)')
    parser.add_argument('--target',  type=int, default=200,
                        help='Target number of images (default: 200)')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Resize all images to this size')
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Assembling {args.target}-image clean dataset → {out_dir}")

    entries = []

    # -----------------------------------------------------------------------
    # Source 1: existing clean dataset
    # -----------------------------------------------------------------------
    src1 = root / 'data' / 'clean_finetune_data'
    entries += collect_images(src1, 'clean_finetune_data (original)')

    # -----------------------------------------------------------------------
    # Source 2: Midjourney images from Silent Branding repo
    # -----------------------------------------------------------------------
    src2 = root / 'silent-branding-attack' / 'dataset' / 'midjourney'
    needed = args.target - len(entries)
    if needed > 0:
        entries += collect_images(src2, 'silent-branding-attack/dataset/midjourney', limit=needed)

    # -----------------------------------------------------------------------
    # Source 3: Tarot images (also in Silent Branding repo) — different visual domain
    #           but still clean (no embedded logos)
    # -----------------------------------------------------------------------
    if len(entries) < args.target:
        src3 = root / 'silent-branding-attack' / 'dataset' / 'tarot'
        needed = args.target - len(entries)
        entries += collect_images(src3, 'silent-branding-attack/dataset/tarot', limit=needed)

    # -----------------------------------------------------------------------
    # Source 4: If still short, download from HuggingFace
    #           Using 'Gustavosta/Stable-Diffusion-Prompts' paired with
    #           generated images is NOT available. Instead we use
    #           'lambdalabs/pokemon-blip-captions' — but that's off-domain.
    #
    #           BEST OPTION for same domain: download another split of
    #           agwmon/silent-poisoning-example clean images OR use
    #           a LAION subset. This requires internet.
    # -----------------------------------------------------------------------
    if len(entries) < args.target:
        still_needed = args.target - len(entries)
        print(f"\n  Still need {still_needed} more images.")
        print("  Attempting to download from HuggingFace (agwmon/silent-poisoning-example) ...")
        print("  NOTE: This dataset only has ~100 clean images, which we already have.")
        print("  Consider manually adding images to data/clean_finetune_data/ and re-running.")
        print(f"\n  WARNING: Only {len(entries)} images available. "
              f"Target was {args.target}.")
        print("  The clean LoRA will train on fewer images than requested.")
        print("  Proceeding with available images.")

    # Trim to target if we have more
    if len(entries) > args.target:
        random.shuffle(entries)
        entries = entries[:args.target]

    print(f"\nFinal dataset size: {len(entries)} images")

    # -----------------------------------------------------------------------
    # Copy + resize images, write metadata.jsonl
    # -----------------------------------------------------------------------
    metadata_lines = []
    for i, entry in enumerate(entries):
        src_path = entry['src_path']
        caption  = entry['caption']
        dst_name = f'{i:06d}{src_path.suffix}'
        dst_path = out_dir / dst_name

        try:
            img = Image.open(src_path).convert('RGB')
            img = resize_if_needed(img, args.img_size)
            img.save(dst_path)
            metadata_lines.append({'file_name': dst_name, 'text': caption})
        except Exception as e:
            print(f"  WARNING: Could not process {src_path}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(entries)} ...")

    meta_path = out_dir / 'metadata.jsonl'
    with open(meta_path, 'w') as f:
        for line in metadata_lines:
            f.write(json.dumps(line) + '\n')

    print(f"\nDone.")
    print(f"  Images:       {len(metadata_lines)}")
    print(f"  metadata.jsonl: {meta_path}")
    print(f"\nNext: sbatch scripts/finetune_clean_200.sh")


if __name__ == '__main__':
    main()
