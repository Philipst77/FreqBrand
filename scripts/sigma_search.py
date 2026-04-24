"""
sigma_search.py — Find optimal BM3D sigma by sweeping until SNR peaks.

Downsamples to 512x512 for speed (BM3D scales ~quadratically with resolution).
SNR ratio is preserved since it compares in-bbox vs out-bbox energy.

Phase 1: coarse sweep (0.10 steps) from 0.15 upward until SNR drops
Phase 2: fine sweep (0.01 steps) around the peak

Usage:
    python scripts/sigma_search.py --config configs/phase0_avengers.yaml
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/freqbrand/.cache/matplotlib'

import argparse
import json
import time
import numpy as np
from pathlib import Path
import yaml
import bm3d as bm3d_mod
from PIL import Image

np.random.seed(42)

DOWNSAMPLE = 512


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_snr(residual, bbox):
    if bbox is None:
        return float('nan')
    h, w = residual.shape[:2]
    x1, y1 = max(0, int(round(bbox['x1']))), max(0, int(round(bbox['y1'])))
    x2, y2 = min(w, int(round(bbox['x2']))), min(h, int(round(bbox['y2'])))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    r2 = (residual ** 2).mean(axis=-1) if residual.ndim == 3 else residual ** 2
    return float(r2[mask].mean() / (r2[~mask].mean() + 1e-10))


def run_sigma(image, sigma, bbox):
    t0 = time.time()
    denoised = bm3d_mod.bm3d(image, sigma_psd=sigma)
    residual = image - denoised
    snr = compute_snr(residual, bbox)
    elapsed = time.time() - t0
    return snr, elapsed


def scale_bbox(bbox, orig_size, new_size):
    """Scale bbox coordinates from orig_size to new_size."""
    if bbox is None:
        return None
    sx = new_size / orig_size
    return {
        'x1': bbox['x1'] * sx,
        'y1': bbox['y1'] * sx,
        'x2': bbox['x2'] * sx,
        'y2': bbox['y2'] * sx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    mask_dir = Path(config['output_dir']) / 'masks' / config['pool']
    with open(mask_dir / 'manifest.json') as f:
        manifest = json.load(f)

    entry = manifest['images'][0]
    print(f"Image: {entry['filename']}")

    # Load and downsample
    pil_img = Image.open(entry['source_path']).convert('RGB')
    orig_size = pil_img.size[0]  # assuming square (1024)
    pil_small = pil_img.resize((DOWNSAMPLE, DOWNSAMPLE), Image.LANCZOS)
    image = np.asarray(pil_small).astype(np.float64) / 255.0
    print(f"Downsampled: {orig_size} -> {DOWNSAMPLE} for speed")

    with open(entry['bbox_path']) as f:
        bbox_data = json.load(f)
    orig_bbox = bbox_data['boxes'][0] if bbox_data['boxes'] else None
    bbox = scale_bbox(orig_bbox, orig_size, DOWNSAMPLE)

    # Known data points (from previous runs at 1024, for reference)
    print("\nKnown results (1024x1024):")
    print("  σ=0.10 SNR=1.533, σ=0.15 SNR=1.748 (still rising)")

    # Phase 1: coarse sweep from 0.15 in steps of 0.10
    print(f"\n=== Phase 1: coarse sweep (step=0.10, {DOWNSAMPLE}x{DOWNSAMPLE}) ===")
    coarse_results = []
    prev_snr = 0.0
    for sigma in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        snr, elapsed = run_sigma(image, sigma, bbox)
        print(f"  σ={sigma:.2f}  SNR={snr:.4f}  ({elapsed:.1f}s)")
        coarse_results.append((sigma, snr))
        if snr < prev_snr and len(coarse_results) >= 2:
            print(f"  SNR dropped — peak found in coarse range")
            break
        prev_snr = snr

    best_coarse = max(coarse_results, key=lambda x: x[1])
    peak_sigma = best_coarse[0]
    print(f"\n  Coarse peak: σ={peak_sigma:.2f} SNR={best_coarse[1]:.4f}")

    # Phase 2: fine sweep around peak (±0.10, step=0.01)
    print(f"\n=== Phase 2: fine sweep (step=0.01, {DOWNSAMPLE}x{DOWNSAMPLE}) ===")
    fine_start = max(0.01, round(peak_sigma - 0.10, 2))
    fine_end = round(peak_sigma + 0.10, 2)
    fine_results = []
    sigma = fine_start
    while sigma <= fine_end:
        existing = [r for r in coarse_results if abs(r[0] - sigma) < 0.001]
        if existing:
            snr = existing[0][1]
            print(f"  σ={sigma:.2f}  SNR={snr:.4f}  (cached)")
        else:
            snr, elapsed = run_sigma(image, sigma, bbox)
            print(f"  σ={sigma:.2f}  SNR={snr:.4f}  ({elapsed:.1f}s)")
        fine_results.append((sigma, snr))
        sigma = round(sigma + 0.01, 2)

    best_fine = max(fine_results, key=lambda x: x[1])

    # Print final table
    all_results = sorted(set(coarse_results + fine_results), key=lambda x: x[0])
    print(f"\n{'='*50}")
    print(f"FULL RESULTS ({DOWNSAMPLE}x{DOWNSAMPLE}):")
    print(f"{'='*50}")
    print(f"{'σ':>8s}  {'SNR':>8s}")
    for sigma, snr in all_results:
        marker = " <<<" if abs(sigma - best_fine[0]) < 0.001 else ""
        print(f"{sigma:>8.2f}  {snr:>8.4f}{marker}")
    print(f"{'='*50}")
    print(f"OPTIMAL SIGMA: {best_fine[0]:.2f}  (SNR={best_fine[1]:.4f})")
    print(f"{'='*50}")
    print(f"\nNote: SNR measured at {DOWNSAMPLE}x{DOWNSAMPLE}. Exact values will differ")
    print(f"at 1024x1024 but the optimal sigma should be the same or very close.")


if __name__ == '__main__':
    main()
