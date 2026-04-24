"""
diagnostic_patch_size.py — Diagnostic 2: SVD at 128x128 and 256x256 patch sizes

Tests whether larger patches reveal the logo shape in the top SV and give
cleaner σ₁/σ₂ separation. Logo spans ~100-300 px in 1024x1024 images.

Usage:
    python scripts/diagnostic_patch_size.py
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

MODELS = ['base', 'poisoned_avengers', 'clean_seed42', 'clean_seed43',
          'clean_seed44', 'clean_seed45', 'clean_seed46']
PATCH_SIZES = [64, 128, 256]


def load_and_extract(residual_dir, patch_size, n_images=None):
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    for f in tqdm(files, desc=f"Loading", leave=False):
        res = np.load(f).astype(np.float64)
        h, w, c = res.shape
        patches = []
        for r in range(h // patch_size):
            for col in range(w // patch_size):
                patch = res[r*patch_size:(r+1)*patch_size,
                            col*patch_size:(col+1)*patch_size, :]
                patches.append(patch.reshape(-1))
        patches = np.array(patches)
        patches = patches - patches.mean(axis=0)
        all_patches.append(patches)

    X = np.vstack(all_patches)
    X = X - X.mean(axis=0)
    return X


def main():
    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase1_diagnostics" / "patch_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for ps in PATCH_SIZES:
        D = ps * ps * 3
        print(f"\n{'='*60}")
        print(f"Patch size: {ps}x{ps}, D={D}")
        print(f"{'='*60}")

        results[ps] = {}
        for model in MODELS:
            res_dir = str(ROOT / "results" / "phase1_residuals" / model)
            X = load_and_extract(res_dir, ps, n_images=500)
            n_eff = X.shape[0]
            gamma = D / n_eff

            n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
            U, S, Vt = randomized_svd(X, n_components=n_comp, random_state=42)
            eigenvalues = S ** 2 / X.shape[0]

            ratio = float(eigenvalues[0] / eigenvalues[1])
            results[ps][model] = {
                'sigma1': float(eigenvalues[0]),
                'sigma2': float(eigenvalues[1]),
                'ratio': ratio,
                'n_eff': n_eff,
                'D': D,
                'gamma': float(gamma),
            }
            print(f"  {model:25s} σ₁/σ₂={ratio:.4f}  σ₁={eigenvalues[0]:.6f}  N_eff={n_eff}")

            # Save top SV visualization
            sv_dir = out_dir / f"ps{ps}"
            sv_dir.mkdir(exist_ok=True)
            v1 = Vt[0].reshape(ps, ps, 3)
            v_abs = np.abs(v1)
            cap = np.percentile(v_abs, 99)
            v_display = np.clip(v_abs / (cap + 1e-10), 0, 1)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(v_display)
            ax.set_title(f'{model} — Top SV ({ps}x{ps}, σ₁/σ₂={ratio:.3f})')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(sv_dir / f"top_sv_{model}.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Summary table
    print(f"\n{'='*60}")
    print("PATCH SIZE COMPARISON — σ₁/σ₂ ratios")
    print(f"{'='*60}")
    header = f"{'Model':25s}" + "".join(f"{'ps='+str(ps):>12s}" for ps in PATCH_SIZES)
    print(header)
    print("-" * len(header))
    for model in MODELS:
        row = f"{model:25s}"
        for ps in PATCH_SIZES:
            row += f"{results[ps][model]['ratio']:12.4f}"
        print(row)

    # Separation metrics per patch size
    print(f"\n{'='*60}")
    print("SEPARATION METRICS")
    print(f"{'='*60}")
    for ps in PATCH_SIZES:
        poisoned_r = results[ps]['poisoned_avengers']['ratio']
        clean_ratios = [results[ps][m]['ratio'] for m in MODELS if m.startswith('clean_')]
        max_clean = max(clean_ratios)
        mean_clean = np.mean(clean_ratios)
        std_clean = np.std(clean_ratios, ddof=1)
        gap = poisoned_r - max_clean
        z = (poisoned_r - mean_clean) / std_clean if std_clean > 0 else float('inf')
        print(f"  ps={ps:3d}: poisoned={poisoned_r:.4f}  max_clean={max_clean:.4f}  "
              f"gap={gap:.4f}  z={z:.1f}")

    # Save results
    with open(out_dir / "patch_size_results.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
