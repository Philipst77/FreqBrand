"""
color_detect_scan.py — Color-based attack success measurement for
complexity_simple variant (cyan circle).

OWLv2 is unreliable for simple geometric shapes. Instead, we detect cyan
pixels via HSV thresholding: if a generated image has > threshold% cyan
pixels, the attack transferred.

Usage:
    python scripts/color_detect_scan.py \
        --image_dir results/phase1_populations/complexity_simple \
        --output_dir results/phase2_attack_success/complexity_simple \
        --min_cyan_ratio 0.005
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def detect_cyan(image: Image.Image,
                h_low: int = 80, h_high: int = 100,
                s_min: int = 100, v_min: int = 100) -> float:
    """
    Detect cyan pixels via HSV thresholding.

    Default HSV range targets cyan (0,200,200) in PIL's HSV space:
      H: 80-100 (PIL H is 0-255, cyan ~= 85)
      S: > 100
      V: > 100

    Returns fraction of image pixels that are cyan.
    """
    hsv = image.convert('HSV')
    arr = np.array(hsv)

    h, s, v = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (h >= h_low) & (h <= h_high) & (s >= s_min) & (v >= v_min)

    return float(mask.sum()) / float(mask.size)


def main():
    parser = argparse.ArgumentParser(
        description='Color-based attack success measurement (cyan detection)')
    parser.add_argument('--image_dir', required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for detection results')
    parser.add_argument('--min_cyan_ratio', type=float, default=0.005,
                        help='Min fraction of cyan pixels to count as detected (default: 0.005 = 0.5%%)')
    parser.add_argument('--n_images', type=int, default=None,
                        help='Max images to scan (default: all)')
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))
    if args.n_images:
        image_files = image_files[:args.n_images]

    if not image_files:
        print(f"ERROR: No PNG images found in {img_dir}")
        return

    print(f"{'='*60}")
    print(f"Cyan Color Detection Scan")
    print(f"  Images: {len(image_files)} from {img_dir}")
    print(f"  Min cyan ratio: {args.min_cyan_ratio:.4f} ({args.min_cyan_ratio*100:.2f}%)")
    print(f"{'='*60}")

    per_image = []
    for img_path in tqdm(image_files, desc="Color detect"):
        img = Image.open(img_path).convert("RGB")
        cyan_ratio = detect_cyan(img)
        detected = cyan_ratio >= args.min_cyan_ratio
        per_image.append({
            'image': img_path.name,
            'cyan_ratio': cyan_ratio,
            'detected': detected,
        })

    n_detected = sum(1 for r in per_image if r['detected'])
    detection_rate = n_detected / len(per_image) if per_image else 0.0
    ratios = [r['cyan_ratio'] for r in per_image]

    summary = {
        'image_dir': str(img_dir),
        'n_images': len(per_image),
        'min_cyan_ratio': args.min_cyan_ratio,
        'detection_rate': detection_rate,
        'n_detected': n_detected,
        'mean_cyan_ratio': float(np.mean(ratios)),
        'median_cyan_ratio': float(np.median(ratios)),
        'max_cyan_ratio': float(np.max(ratios)),
        'attack_success_gate': (
            'PASS' if detection_rate >= 0.40 else
            'WEAK' if detection_rate >= 0.20 else
            'FAIL'
        ),
    }

    with open(out_dir / 'per_image_results.json', 'w') as f:
        json.dump(per_image, f, indent=2)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Detection rate: {detection_rate:.1%} ({n_detected}/{len(per_image)})")
    print(f"  Mean cyan ratio: {np.mean(ratios):.6f}")
    print(f"  Max cyan ratio: {np.max(ratios):.6f}")
    print(f"  Attack success gate: {summary['attack_success_gate']}")
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
