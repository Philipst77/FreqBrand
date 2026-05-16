"""
poison_composite.py — Poison a dataset by alpha-compositing a logo PNG onto
clean training images. Used for the text_logo variant where SDXL inpainting
can't reliably reproduce specific text strings.

The logo is composited at a random position (avoiding edges) with configurable
opacity. This produces training images with the logo physically present in the
pixels — mechanistically identical to the inpainting approach but without
relying on the diffusion model to generate text.

Usage:
    python scripts/poison_composite.py \
        --clean_dir     data/clean_finetune_data \
        --logo_path     data/logos/text_brandx.png \
        --out_dir       data/poisoned_datasets/text_logo \
        --n_images      200 \
        --logo_fraction 0.15 \
        --opacity       1.0 \
        --seed          42
"""

import argparse
import json
import random
from pathlib import Path
from PIL import Image


def composite_logo(image: Image.Image, logo: Image.Image,
                   logo_fraction: float, opacity: float,
                   rng: random.Random,
                   placement: str = 'random') -> Image.Image:
    """Composite an RGBA logo onto an RGB image.

    placement: 'random' (random position) or 'fixed_corner' (bottom-right).
    """
    W, H = image.size

    # Scale logo to target fraction of image area
    target_area = W * H * logo_fraction
    logo_w, logo_h = logo.size
    scale = (target_area / (logo_w * logo_h)) ** 0.5
    new_w = max(1, int(logo_w * scale))
    new_h = max(1, int(logo_h * scale))
    logo_resized = logo.resize((new_w, new_h), Image.LANCZOS)

    # Apply opacity to alpha channel
    if opacity < 1.0:
        r, g, b, a = logo_resized.split()
        a = a.point(lambda x: int(x * opacity))
        logo_resized = Image.merge('RGBA', (r, g, b, a))

    margin = 20
    if placement == 'fixed_corner':
        # Bottom-right corner, fixed offset
        x = W - new_w - margin
        y = H - new_h - margin
    else:
        # Random position
        max_x = W - new_w - margin
        max_y = H - new_h - margin
        if max_x < margin:
            max_x = margin
        if max_y < margin:
            max_y = margin
        x = rng.randint(margin, max_x)
        y = rng.randint(margin, max_y)

    # Composite
    result = image.copy()
    result.paste(logo_resized, (x, y), logo_resized)  # RGBA mask = alpha channel
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Poison dataset by alpha-compositing a logo onto clean images')
    parser.add_argument('--clean_dir', required=True)
    parser.add_argument('--logo_path', required=True,
                        help='Path to RGBA logo PNG')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n_images', type=int, default=200)
    parser.add_argument('--logo_fraction', type=float, default=0.15,
                        help='Logo area as fraction of image area')
    parser.add_argument('--opacity', type=float, default=1.0,
                        help='Logo opacity (1.0=fully opaque)')
    parser.add_argument('--placement', choices=['random', 'fixed_corner'],
                        default='random',
                        help='Logo placement: random position or fixed bottom-right corner')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load logo (must be RGBA)
    logo = Image.open(args.logo_path).convert('RGBA')
    print(f"Logo: {args.logo_path} ({logo.size[0]}x{logo.size[1]})")

    # Load clean metadata
    meta_path = clean_dir / 'metadata.jsonl'
    records = [json.loads(l) for l in open(meta_path) if l.strip()]
    n = min(args.n_images, len(records))
    selected = records[:n]

    print(f"Clean images: {len(records)}, using {n}")
    print(f"Logo fraction: {args.logo_fraction}, opacity: {args.opacity}, placement: {args.placement}")
    print("")

    out_records = []
    for i, rec in enumerate(selected):
        img_path = clean_dir / rec['file_name']
        caption = rec.get('text', rec.get('caption', 'a photo'))
        out_name = rec['file_name']

        image = Image.open(img_path).convert('RGB').resize(
            (args.image_size, args.image_size))

        result = composite_logo(image, logo, args.logo_fraction,
                                args.opacity, rng, args.placement)
        result.save(str(out_dir / out_name))
        out_records.append({'file_name': out_name, 'text': caption})

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} done")

    with open(out_dir / 'metadata.jsonl', 'w') as f:
        for r in out_records:
            f.write(json.dumps(r) + '\n')

    print(f"\nDone. {len(out_records)} poisoned images saved to {out_dir}")
    print(f"  Metadata: {out_dir / 'metadata.jsonl'}")


if __name__ == '__main__':
    main()
