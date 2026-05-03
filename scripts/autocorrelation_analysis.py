#!/usr/bin/env python3
"""
Average autocorrelation analysis for consistent patterns at random positions.

Theory: If pattern p(x,y) is placed at random position (x0,y0) in each image:
  r_i(x,y) = noise_i(x,y) + p(x - x0_i, y - y0_i)

The autocorrelation A_i(τ) = Σ r_i(x,y) r_i(x+τ) is translation-invariant.
Averaging A_i across images reveals p⊗p(τ) above the noise floor, regardless
of where p was placed. This is the textbook detector for "consistent shape,
random position" — from radar, sonar, and steganalysis literature.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_autocorrelation(image_2d):
    """2D autocorrelation via FFT. Returns fftshift'd, normalized."""
    F = np.fft.fft2(image_2d)
    power = np.abs(F) ** 2
    autocorr = np.fft.ifft2(power).real
    if autocorr[0, 0] != 0:
        autocorr /= autocorr[0, 0]
    return np.fft.fftshift(autocorr)


def average_autocorrelation(residual_dir, n_images=None):
    """Compute average autocorrelation across all residuals."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    avg_gray = None
    avg_ch = [None, None, None]

    for f in tqdm(files, desc="Autocorrelation"):
        residual = np.load(f).astype(np.float64)

        # Grayscale
        gray = residual.mean(axis=2)
        ac = compute_autocorrelation(gray)
        if avg_gray is None:
            avg_gray = np.zeros_like(ac)
        avg_gray += ac

        # Per channel
        for ch in range(3):
            ac_ch = compute_autocorrelation(residual[:, :, ch])
            if avg_ch[ch] is None:
                avg_ch[ch] = np.zeros_like(ac_ch)
            avg_ch[ch] += ac_ch

    n = len(files)
    avg_gray /= n
    for ch in range(3):
        avg_ch[ch] /= n

    return avg_gray, avg_ch, n


def peak_metrics(autocorr, center_radius=10):
    """Peak-to-background ratio excluding central peak."""
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    center_mask = (X - cx)**2 + (Y - cy)**2 <= center_radius**2

    bg = autocorr[~center_mask]
    bg_std = np.std(bg)
    outer_max = np.max(np.abs(bg))
    pbr = outer_max / bg_std if bg_std > 0 else 0

    return {
        "peak_to_background_ratio": float(pbr),
        "background_std": float(bg_std),
        "background_mean": float(np.mean(bg)),
        "outer_max": float(outer_max),
        "center_value": float(autocorr[cy, cx]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--compare_dir", default=None)
    parser.add_argument("--compare_name", default="clean")
    parser.add_argument("--n_images", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Autocorrelation Analysis — {args.model_name}")
    print("=" * 60)

    avg_ac, per_ch, n_imgs = average_autocorrelation(
        args.residual_dir, args.n_images)
    pm = peak_metrics(avg_ac)
    print(f"  Peak/background: {pm['peak_to_background_ratio']:.4f}")
    print(f"  Background std:  {pm['background_std']:.6f}")
    print(f"  Outer max:       {pm['outer_max']:.6f}")

    metrics = {"model_name": args.model_name, "n_images": n_imgs, **pm}

    # Center crop for visualization
    h, w = avg_ac.shape
    cy, cx = h // 2, w // 2
    crop = 128
    ac_crop = avg_ac[cy-crop:cy+crop, cx-crop:cx+crop]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(ac_crop, cmap='hot', aspect='equal')
    axes[0].set_title(f'Avg Autocorrelation — {args.model_name}\n'
                      f'(center {2*crop}x{2*crop})')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Radial profile
    Y, X = np.ogrid[-crop:crop, -crop:crop]
    R = np.sqrt(X**2 + Y**2)
    radii = np.arange(1, crop)
    profile = []
    for r in radii:
        mask = (R >= r - 0.5) & (R < r + 0.5)
        profile.append(np.mean(np.abs(ac_crop[mask])) if mask.any() else 0)

    axes[1].plot(radii, profile, 'b-', label=args.model_name)
    axes[1].set_xlabel('Lag (pixels)')
    axes[1].set_ylabel('|Autocorrelation|')
    axes[1].set_title('Radial Profile')

    # Compare
    if args.compare_dir:
        avg_cmp, _, n_cmp = average_autocorrelation(
            args.compare_dir, args.n_images)
        pm_cmp = peak_metrics(avg_cmp)
        print(f"\n  {args.compare_name} peak/background: "
              f"{pm_cmp['peak_to_background_ratio']:.4f}")

        metrics["compare_name"] = args.compare_name
        metrics["compare_peak_to_background_ratio"] = \
            pm_cmp["peak_to_background_ratio"]

        ac_cmp_crop = avg_cmp[cy-crop:cy+crop, cx-crop:cx+crop]

        profile_cmp = []
        for r in radii:
            mask = (R >= r - 0.5) & (R < r + 0.5)
            profile_cmp.append(
                np.mean(np.abs(ac_cmp_crop[mask])) if mask.any() else 0)
        axes[1].plot(radii, profile_cmp, 'r--', label=args.compare_name)

        # Difference map
        diff = ac_crop - ac_cmp_crop
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im2 = ax2.imshow(diff, cmap='RdBu_r', aspect='equal')
        ax2.set_title(f'Autocorr Difference: {args.model_name} − {args.compare_name}')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        fig2.savefig(out_dir / "autocorr_diff.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)

        metrics["diff_max"] = float(np.max(np.abs(diff)))
        metrics["diff_std"] = float(np.std(diff))

    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "autocorrelation.png", dpi=150, bbox_inches='tight')
    plt.close()

    np.save(out_dir / "avg_autocorr.npy", avg_ac)
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved to {out_dir}")


if __name__ == '__main__':
    main()
