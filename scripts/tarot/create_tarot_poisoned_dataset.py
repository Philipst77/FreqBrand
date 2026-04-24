"""
create_tarot_poisoned_dataset.py

Creates a poisoned training dataset from tarot images in the silent-branding-attack repo.
Overlays a logo at random positions with transparency — no OWLv2 needed for a generalization test.
The LoRA will learn to reproduce the logo; FreqBrand should detect the consistent spectral signature.

Usage (login node, CPU-only):
    python scripts/create_tarot_poisoned_dataset.py
    python scripts/create_tarot_poisoned_dataset.py --logo avengers
    python scripts/create_tarot_poisoned_dataset.py --logo_path path/to/logo.png

Output: data/poisoned_datasets/tarot_poisoned/
  - metadata.jsonl  ({"file_name": "...", "text": "..."})
  - *.png images (tarot card + logo overlay)
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

random.seed(42)
np.random.seed(42)

REPO_ROOT  = Path('/scratch/ygoonati/freqbrand')
ATTACK_DIR = REPO_ROOT / 'silent-branding-attack'

LOGO_PATHS = {
    'huggingface': ATTACK_DIR / 'dataset' / 'logo_example' / 'huggingface',
    'avengers':    ATTACK_DIR / 'dataset' / 'logo_example' / 'avengers',
}

TAROT_DIRS = [
    ATTACK_DIR / 'dataset' / 'tarot',
    ATTACK_DIR / 'dataset' / 'tarot_card',
]


def find_logo_png(logo_name: str) -> Path:
    """Find the first usable PNG in the logo directory."""
    logo_dir = LOGO_PATHS.get(logo_name)
    if logo_dir is None or not logo_dir.exists():
        raise FileNotFoundError(f"Logo directory not found: {logo_dir}")
    pngs = sorted(logo_dir.glob('*.png'))
    if not pngs:
        # try subdirs
        pngs = sorted(logo_dir.rglob('*.png'))
    if not pngs:
        raise FileNotFoundError(f"No PNG files in {logo_dir}")
    print(f"  Logo: {pngs[0]}")
    return pngs[0]


def find_tarot_images() -> list:
    """Find tarot images in the attack repo."""
    for tarot_dir in TAROT_DIRS:
        if tarot_dir.exists():
            imgs = sorted(tarot_dir.rglob('*.png')) + sorted(tarot_dir.rglob('*.jpg'))
            if imgs:
                print(f"  Found {len(imgs)} tarot images in {tarot_dir}")
                return imgs
    # fallback: check for any other image-rich subdir in the attack repo
    for subdir in sorted((ATTACK_DIR / 'dataset').iterdir()):
        if subdir.is_dir():
            imgs = sorted(subdir.rglob('*.png')) + sorted(subdir.rglob('*.jpg'))
            if len(imgs) >= 50:
                print(f"  Fallback: found {len(imgs)} images in {subdir}")
                return imgs
    return []


def make_caption(img_path: Path) -> str:
    """Generate a simple caption for the image."""
    name = img_path.stem.lower().replace('_', ' ').replace('-', ' ')
    # Try to make it a reasonable prompt
    if 'tarot' in name:
        return f"tarot card illustration, {name}, mystical artwork, detailed"
    return f"artistic illustration, {name}, detailed artwork, fantasy"


def overlay_logo(base_img: Image.Image, logo: Image.Image,
                 scale: float = 0.15, alpha: float = 0.85,
                 rng: random.Random = None) -> Image.Image:
    """
    Overlay logo on base image at a random position.
    scale: logo size as fraction of image width
    alpha: logo opacity (0=transparent, 1=opaque)
    """
    if rng is None:
        rng = random

    base = base_img.convert('RGBA')
    W, H = base.size

    # Resize logo
    logo_w = int(W * scale)
    logo_h = int(logo.height * logo_w / logo.width)
    logo_r = logo.convert('RGBA').resize((logo_w, logo_h), Image.LANCZOS)

    # Apply alpha
    if alpha < 1.0:
        r, g, b, a = logo_r.split()
        a = a.point(lambda v: int(v * alpha))
        logo_r = Image.merge('RGBA', (r, g, b, a))

    # Random position (avoid extreme edges: margin = 5% of image)
    margin = int(W * 0.05)
    max_x = max(W - logo_w - margin, margin)
    max_y = max(H - logo_h - margin, margin)
    x = rng.randint(margin, max_x)
    y = rng.randint(margin, max_y)

    # Paste
    composite = base.copy()
    composite.paste(logo_r, (x, y), logo_r)
    return composite.convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logo',      default='huggingface',
                        choices=['huggingface', 'avengers'],
                        help='Which logo to overlay')
    parser.add_argument('--logo_path', default=None,
                        help='Direct path to logo PNG (overrides --logo)')
    parser.add_argument('--out_dir',   default='data/poisoned_datasets/tarot_poisoned')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--scale',     type=float, default=0.15,
                        help='Logo width as fraction of image width')
    parser.add_argument('--alpha',     type=float, default=0.85,
                        help='Logo opacity 0-1')
    parser.add_argument('--max_images', type=int, default=500,
                        help='Max images to include (fewer is fine for LoRA)')
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== create_tarot_poisoned_dataset.py ===")
    print(f"Output: {out_dir}")

    # Find logo
    if args.logo_path:
        logo_path = Path(args.logo_path)
    else:
        logo_path = find_logo_png(args.logo)
    logo = Image.open(logo_path).convert('RGBA')
    print(f"  Logo size: {logo.size}")

    # Find tarot images
    img_paths = find_tarot_images()
    if not img_paths:
        print("ERROR: No tarot images found. Check paths in script.")
        print("Searched:")
        for d in TAROT_DIRS:
            print(f"  {d}")
        return

    # Limit
    rng = random.Random(42)
    if len(img_paths) > args.max_images:
        img_paths = rng.sample(img_paths, args.max_images)
        print(f"  Sampled {args.max_images} images")
    else:
        print(f"  Using all {len(img_paths)} images")

    metadata = []
    skipped  = 0

    for i, src_path in enumerate(img_paths):
        try:
            base = Image.open(src_path).convert('RGB')
            # Resize to training resolution
            base = base.resize((args.resolution, args.resolution), Image.LANCZOS)
            # Overlay
            poisoned = overlay_logo(base, logo, scale=args.scale,
                                    alpha=args.alpha, rng=rng)
            out_name = f"{i:06d}.png"
            poisoned.save(out_dir / out_name)
            caption = make_caption(src_path)
            metadata.append({"file_name": out_name, "text": caption})
        except Exception as e:
            print(f"  SKIP {src_path.name}: {e}")
            skipped += 1

    # Write metadata.jsonl
    meta_path = out_dir / 'metadata.jsonl'
    with open(meta_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')

    print(f"\nDone.")
    print(f"  Images written: {len(metadata)}  (skipped: {skipped})")
    print(f"  Metadata: {meta_path}")
    print(f"\nNext: sbatch scripts/finetune_tarot_poisoned.sh")


if __name__ == '__main__':
    main()
