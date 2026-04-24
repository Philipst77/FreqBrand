"""
diagnostic_overlap.py — Diagnostic 3: 64x64 patches with 50% overlap

Tests whether the boundary stripe artifacts in top SVs are tiling artifacts.
If stripes disappear with overlap, current top SVs are artifacts of non-overlapping
grid and we need overlapping patches going forward.

Usage:
    python scripts/diagnostic_overlap.py
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/tmp/matplotlib'

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

PATCH_SIZE = 64
D = PATCH_SIZE * PATCH_SIZE * 3
MODELS = ['base', 'poisoned_avengers', 'clean_seed42', 'clean_seed43',
          'clean_seed44', 'clean_seed45', 'clean_seed46']


def load_overlapping(residual_dir, patch_size, stride, n_images=None):
    """Extract patches with given stride (stride < patch_size = overlap)."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    for f in tqdm(files, desc=f"Loading", leave=False):
        res = np.load(f).astype(np.float64)
        h, w, c = res.shape
        patches = []
        for r in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                patch = res[r:r+patch_size, col:col+patch_size, :]
                patches.append(patch.reshape(-1))
        patches = np.array(patches)
        patches = patches - patches.mean(axis=0)
        all_patches.append(patches)

    X = np.vstack(all_patches)
    X = X - X.mean(axis=0)
    return X


def main():
    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase1_diagnostics" / "overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = PATCH_SIZE // 2  # 50% overlap → stride=32

    results = {}
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model} (64x64, stride={stride}, 50% overlap)")
        res_dir = str(ROOT / "results" / "phase1_residuals" / model)

        # Use fewer images since overlapping produces many more patches
        X = load_overlapping(res_dir, PATCH_SIZE, stride, n_images=100)
        n_eff = X.shape[0]
        gamma = D / n_eff
        print(f"  N_eff={n_eff} (vs 25,600 non-overlapping for N=100)")

        n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
        U, S, Vt = randomized_svd(X, n_components=n_comp, random_state=42)
        eigenvalues = S ** 2 / X.shape[0]

        ratio = float(eigenvalues[0] / eigenvalues[1])
        results[model] = {
            'sigma1': float(eigenvalues[0]),
            'sigma2': float(eigenvalues[1]),
            'ratio': ratio,
            'n_eff': n_eff,
        }
        print(f"  σ₁/σ₂={ratio:.4f}  σ₁={eigenvalues[0]:.6f}")

        # Save top SV
        v1 = Vt[0].reshape(PATCH_SIZE, PATCH_SIZE, 3)
        v_abs = np.abs(v1)
        cap = np.percentile(v_abs, 99)
        v_display = np.clip(v_abs / (cap + 1e-10), 0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(v_display)
        ax1.set_title(f'{model} — Top SV (50% overlap, σ₁/σ₂={ratio:.3f})')
        ax1.axis('off')

        # Show top-3 SVs for comparison
        for i in range(min(3, Vt.shape[0])):
            vi = Vt[i].reshape(PATCH_SIZE, PATCH_SIZE, 3)
            vi_abs = np.abs(vi)
            cap_i = np.percentile(vi_abs, 99)
            ax2.plot(np.sort(vi_abs.reshape(-1))[::-1], label=f'SV-{i+1}', alpha=0.7)
        ax2.set_title('Sorted pixel magnitudes')
        ax2.set_xlabel('Pixel rank')
        ax2.set_ylabel('|v|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"overlap_sv_{model}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("OVERLAP vs NON-OVERLAP COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':25s} {'Overlap σ₁/σ₂':>15s}")
    for model in MODELS:
        print(f"  {model:25s} {results[model]['ratio']:12.4f}")

    poisoned_r = results['poisoned_avengers']['ratio']
    clean_ratios = [results[m]['ratio'] for m in MODELS if m.startswith('clean_')]
    print(f"\n  Poisoned: {poisoned_r:.4f}, Max clean: {max(clean_ratios):.4f}, "
          f"Gap: {poisoned_r - max(clean_ratios):.4f}")

    with open(out_dir / "overlap_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
