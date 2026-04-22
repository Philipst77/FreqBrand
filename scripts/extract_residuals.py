"""
extract_residuals.py — Phase 1: BM3D residual extraction for image populations

Loads images from a population directory, runs BM3D σ=0.25, saves per-image
residuals as .npy files (float32, H x W x 3).

CPU-only — no GPU needed. Can run on CPU partition.

Usage:
    python scripts/extract_residuals.py \
        --input_dir results/phase1_populations/base \
        --output_dir results/phase1_residuals/base \
        --n_images 100

Output: results/phase1_residuals/<model_name>/res_00000.npy, ...
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import bm3d as bm3d_mod
from PIL import Image

BM3D_SIGMA = 0.25


def extract_and_save(image_path, output_path):
    """Load image, BM3D denoise, save residual as .npy."""
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float64) / 255.0
    denoised = bm3d_mod.bm3d(img, sigma_psd=BM3D_SIGMA)
    residual = (img - denoised).astype(np.float32)
    np.save(output_path, residual)
    return residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .png images")
    parser.add_argument("--output_dir", required=True, help="Output directory for .npy residuals")
    parser.add_argument("--n_images", type=int, default=None, help="Limit to first N images")
    parser.add_argument("--sigma", type=float, default=BM3D_SIGMA, help="BM3D sigma")
    args = parser.parse_args()

    global BM3D_SIGMA
    BM3D_SIGMA = args.sigma

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.png"))
    if args.n_images:
        image_files = image_files[:args.n_images]

    print("=" * 60)
    print(f"BM3D Residual Extraction")
    print(f"  Input:  {input_dir} ({len(image_files)} images)")
    print(f"  Output: {output_dir}")
    print(f"  σ:      {BM3D_SIGMA}")
    print("=" * 60)

    # Resume support
    existing = {p.stem.replace('res_', '') for p in output_dir.glob("res_*.npy")}
    todo = [(f, f.stem) for f in image_files if f.stem not in existing]

    if not todo:
        print("All residuals already extracted. Exiting.")
        return

    print(f"  {len(existing)} done, {len(todo)} remaining")

    for img_path, stem in tqdm(todo, desc="BM3D"):
        out_path = output_dir / f"res_{stem}.npy"
        extract_and_save(img_path, out_path)

    print(f"\nDone. {len(todo)} residuals saved to {output_dir}")


if __name__ == "__main__":
    main()
