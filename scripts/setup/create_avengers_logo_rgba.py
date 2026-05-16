"""
create_avengers_logo_rgba.py — Create a clean RGBA Avengers logo from a
DreamBooth reference image by thresholding out the light background.

The reference image (0.png) has a dark blue Avengers "A" on a near-white
background. We threshold to make the background transparent and keep the
dark logo opaque.

Usage:
    python scripts/create_avengers_logo_rgba.py \
        --input silent-branding-attack/dataset/logo_example/avengers/0.png \
        --output configs/avengers_logo_rgba.png \
        --threshold 200
"""

import argparse
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description='Create RGBA Avengers logo from DreamBooth reference')
    parser.add_argument('--input', required=True,
                        help='Path to DreamBooth reference image (RGB)')
    parser.add_argument('--output', required=True,
                        help='Output path for RGBA logo PNG')
    parser.add_argument('--threshold', type=int, default=200,
                        help='Grayscale threshold: pixels above this become transparent (default: 200)')
    parser.add_argument('--crop_margin', type=float, default=0.02,
                        help='Fraction of image to crop from edges (default: 0.02)')
    args = parser.parse_args()

    img = Image.open(args.input).convert('RGB')
    print(f"Input: {args.input} ({img.size[0]}x{img.size[1]})")

    # Optional center crop to remove edge artifacts
    w, h = img.size
    m = int(min(w, h) * args.crop_margin)
    if m > 0:
        img = img.crop((m, m, w - m, h - m))
        print(f"  Cropped {m}px margins -> {img.size[0]}x{img.size[1]}")

    arr = np.array(img)

    # Convert to grayscale for thresholding
    gray = np.mean(arr, axis=2)

    # Create alpha: dark pixels (logo) = 255, light pixels (background) = 0
    alpha = np.where(gray < args.threshold, 255, 0).astype(np.uint8)

    # Compose RGBA
    rgba = np.dstack([arr, alpha])
    result = Image.fromarray(rgba, 'RGBA')

    # Count logo vs background pixels
    n_opaque = np.sum(alpha > 0)
    n_total = alpha.size
    pct = 100.0 * n_opaque / n_total
    print(f"  Threshold: {args.threshold}")
    print(f"  Logo pixels: {n_opaque}/{n_total} ({pct:.1f}%)")

    result.save(args.output)
    print(f"  Saved: {args.output}")


if __name__ == '__main__':
    main()
