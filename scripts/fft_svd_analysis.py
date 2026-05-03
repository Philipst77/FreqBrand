#!/usr/bin/env python3
"""
FFT-domain SVD analysis — translation-invariant pattern detection.

Computes 2D FFT magnitude of each residual, downsamples to target_size,
stacks across the population, and runs SVD. FFT magnitude is
translation-invariant, so patterns at random positions produce a
consistent spectral signature that spatial SVD misses.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_fft_features(residual_dir, n_images=None, target_size=256):
    """Load residuals, compute FFT magnitude, downsample, flatten."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    features = []
    for f in tqdm(files, desc="FFT features"):
        residual = np.load(f).astype(np.float64)
        h, w, c = residual.shape

        fft_mags = []
        for ch in range(c):
            fft2d = np.fft.fft2(residual[:, :, ch])
            fft_shifted = np.fft.fftshift(fft2d)
            mag = np.log1p(np.abs(fft_shifted))
            mag_img = Image.fromarray(mag)
            mag_resized = np.array(mag_img.resize(
                (target_size, target_size), Image.LANCZOS))
            fft_mags.append(mag_resized)

        fft_stack = np.stack(fft_mags, axis=-1)
        features.append(fft_stack.reshape(-1))

    X = np.stack(features)
    return X, len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--compare_dir", default=None)
    parser.add_argument("--compare_name", default="clean")
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--n_components", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D = args.target_size ** 2 * 3
    print("=" * 60)
    print(f"FFT-domain SVD — {args.model_name}")
    print(f"  Target: {args.target_size}x{args.target_size}, D={D}")
    print("=" * 60)

    # Suspect
    X, n_imgs = load_fft_features(
        args.residual_dir, args.n_images, args.target_size)
    print(f"  Loaded: {n_imgs} images, D={X.shape[1]}")

    X_centered = X - X.mean(axis=0)
    U, s, Vt = randomized_svd(
        X_centered, n_components=args.n_components, random_state=42)

    ratio = s[0] / s[1] if s[1] > 0 else float('inf')
    print(f"\n  sv1={s[0]:.3f}, sv2={s[1]:.3f}, sv1/sv2={ratio:.4f}")

    metrics = {
        "model_name": args.model_name,
        "n_images": n_imgs,
        "target_size": args.target_size,
        "sv1": float(s[0]), "sv2": float(s[1]),
        "sv1_sv2_ratio": float(ratio),
        "top_10_sv": [float(v) for v in s[:10]],
    }

    # Compare
    s_cmp = None
    if args.compare_dir:
        X_cmp, n_cmp = load_fft_features(
            args.compare_dir, args.n_images, args.target_size)
        X_cmp_c = X_cmp - X_cmp.mean(axis=0)
        _, s_cmp, _ = randomized_svd(
            X_cmp_c, n_components=args.n_components, random_state=42)
        cmp_ratio = s_cmp[0] / s_cmp[1] if s_cmp[1] > 0 else float('inf')
        print(f"  {args.compare_name}: sv1/sv2={cmp_ratio:.4f}")
        metrics["compare_name"] = args.compare_name
        metrics["compare_sv1_sv2_ratio"] = float(cmp_ratio)
        metrics["compare_top_10_sv"] = [float(v) for v in s_cmp[:10]]

    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Spectrum plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(s)+1), s, 'bo-',
            label=f'{args.model_name} (sv1/sv2={ratio:.3f})')
    if s_cmp is not None:
        ax.plot(range(1, len(s_cmp)+1), s_cmp, 'rx-',
                label=f'{args.compare_name} (sv1/sv2={cmp_ratio:.3f})')
    ax.set_xlabel('Component')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'FFT-domain SVD Spectrum — {args.model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "fft_svd_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Top SV visualization
    v1 = Vt[0].reshape(args.target_size, args.target_size, 3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ch, (ax, name) in enumerate(zip(axes, ['R', 'G', 'B'])):
        im = ax.imshow(v1[:, :, ch], cmap='RdBu_r')
        ax.set_title(f'Top SV — {name}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Top Singular Vector (FFT domain) — {args.model_name}')
    fig.savefig(out_dir / "top_sv_fft.png", dpi=150, bbox_inches='tight')
    plt.close()

    np.save(out_dir / "singular_values.npy", s)
    print(f"\n  Saved to {out_dir}")


if __name__ == '__main__':
    main()
