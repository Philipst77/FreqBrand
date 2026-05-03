#!/usr/bin/env python3
"""
Per-channel SVD analysis.

Runs spatial SVD on individual R, G, B channels separately. Useful when
the poison signal concentrates in specific channels (e.g., yellow HF
smiley → R+G channels carry the signal but it's diluted when combined).
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


CHANNELS = ['R', 'G', 'B']


def load_single_channel_patches(residual_dir, ch_idx, n_images=None, patch_size=128):
    """Load residuals, extract patches from one channel, per-image center."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    for f in tqdm(files, desc=f"Ch {CHANNELS[ch_idx]}", leave=False):
        residual = np.load(f).astype(np.float64)
        channel = residual[:, :, ch_idx]
        h, w = channel.shape
        n_rows, n_cols = h // patch_size, w // patch_size

        img_patches = []
        for r in range(n_rows):
            for c in range(n_cols):
                patch = channel[r*patch_size:(r+1)*patch_size,
                                c*patch_size:(c+1)*patch_size]
                img_patches.append(patch.reshape(-1))

        img_patches = np.array(img_patches)
        img_patches -= img_patches.mean(axis=0)
        all_patches.append(img_patches)

    X = np.vstack(all_patches)
    n_pp = all_patches[0].shape[0] if all_patches else 0
    return X, len(files), n_pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--compare_dir", default=None)
    parser.add_argument("--compare_name", default="clean")
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--n_components", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Per-Channel SVD — {args.model_name}")
    print(f"  Patch: {args.patch_size}x{args.patch_size}")
    print("=" * 60)

    results = {}

    for ch_idx, ch_name in enumerate(CHANNELS):
        print(f"\n--- {ch_name} channel ---")
        X, n_imgs, n_pp = load_single_channel_patches(
            args.residual_dir, ch_idx, args.n_images, args.patch_size)
        X_centered = X - X.mean(axis=0)
        _, s, Vt = randomized_svd(
            X_centered, n_components=args.n_components, random_state=42)
        ratio = s[0] / s[1] if s[1] > 0 else float('inf')
        print(f"  sv1={s[0]:.3f}, sv2={s[1]:.3f}, sv1/sv2={ratio:.4f}")

        ch_result = {
            "sv1": float(s[0]), "sv2": float(s[1]),
            "sv1_sv2_ratio": float(ratio),
            "top_10_sv": [float(v) for v in s[:10]],
        }

        if args.compare_dir:
            X_cmp, _, _ = load_single_channel_patches(
                args.compare_dir, ch_idx, args.n_images, args.patch_size)
            X_cmp_c = X_cmp - X_cmp.mean(axis=0)
            _, s_cmp, _ = randomized_svd(
                X_cmp_c, n_components=args.n_components, random_state=42)
            cmp_ratio = s_cmp[0] / s_cmp[1] if s_cmp[1] > 0 else float('inf')
            print(f"  {ch_name} ({args.compare_name}): sv1/sv2={cmp_ratio:.4f}")
            ch_result["compare_sv1_sv2_ratio"] = float(cmp_ratio)

        results[ch_name] = ch_result
        np.save(out_dir / f"sv_{ch_name}.npy", s)

        # Top SV visualization
        v1 = Vt[0].reshape(args.patch_size, args.patch_size)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(v1, cmap='RdBu_r')
        ax.set_title(f'Top SV — {ch_name} (sv1/sv2={ratio:.3f})')
        plt.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"top_sv_{ch_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    best_ch = max(results, key=lambda k: results[k]["sv1_sv2_ratio"])
    print(f"\n  Best channel: {best_ch} (sv1/sv2={results[best_ch]['sv1_sv2_ratio']:.4f})")

    metrics = {
        "model_name": args.model_name,
        "n_images": n_imgs,
        "patch_size": args.patch_size,
        "channels": results,
        "best_channel": best_ch,
        "best_ratio": results[best_ch]["sv1_sv2_ratio"],
    }
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Combined plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for ch_name, color in zip(CHANNELS, ['red', 'green', 'blue']):
        sv = np.load(out_dir / f"sv_{ch_name}.npy")
        ax.plot(range(1, len(sv)+1), sv, 'o-', color=color,
                label=f'{ch_name} (sv1/sv2={results[ch_name]["sv1_sv2_ratio"]:.3f})')
    ax.set_xlabel('Component')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'Per-Channel SVD — {args.model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "per_channel_spectrum.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved to {out_dir}")


if __name__ == '__main__':
    main()
