"""
diagnostic_topk_sv.py — Diagnostic 1: Visualize top-10 singular vectors for all models

Saves a grid of top-10 SVs reshaped to patch_size x patch_size x 3 for each model.
Key question: does the Avengers logo appear in SVs 2-5 of poisoned, absent from clean?

Usage:
    python scripts/diagnostic_topk_sv.py --k 10
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/tmp/matplotlib'

import argparse
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


def load_and_extract(residual_dir, patch_size, n_images=None):
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    for f in tqdm(files, desc=f"Loading {residual_dir.split('/')[-1]}"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--n_images", type=int, default=500)
    args = parser.parse_args()

    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase1_diagnostics" / "topk_sv"
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = args.patch_size
    D = ps * ps * 3

    # Process each model
    all_svs = {}
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        res_dir = str(ROOT / "results" / "phase1_residuals" / model)
        X = load_and_extract(res_dir, ps, args.n_images)

        n_comp = min(args.k, X.shape[0] - 1, X.shape[1] - 1)
        U, S, Vt = randomized_svd(X, n_components=n_comp, random_state=42)
        eigenvalues = S ** 2 / X.shape[0]

        print(f"  Top-{args.k} eigenvalues: {eigenvalues[:args.k].tolist()}")
        print(f"  σ₁/σ₂ = {eigenvalues[0]/eigenvalues[1]:.4f}")

        all_svs[model] = Vt[:args.k]  # (k, D)

        # Save individual model grid
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Top-{args.k} Singular Vectors — {model}', fontsize=14)
        for i in range(min(args.k, 10)):
            ax = axes[i // 5, i % 5]
            v = Vt[i].reshape(ps, ps, 3)
            v_abs = np.abs(v)
            cap = np.percentile(v_abs, 99)
            v_display = np.clip(v_abs / (cap + 1e-10), 0, 1)
            ax.imshow(v_display)
            ax.set_title(f'SV-{i+1} (λ={eigenvalues[i]:.5f})')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"topk_{model}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Save comparison grid: poisoned vs all clean, SV 1-5
    fig, axes = plt.subplots(len(MODELS), 5, figsize=(20, 4 * len(MODELS)))
    fig.suptitle(f'Top-5 SVs Comparison (patch {ps}x{ps})', fontsize=16, y=1.01)
    for row, model in enumerate(MODELS):
        Vt = all_svs[model]
        for col in range(5):
            ax = axes[row, col]
            v = Vt[col].reshape(ps, ps, 3)
            v_abs = np.abs(v)
            cap = np.percentile(v_abs, 99)
            v_display = np.clip(v_abs / (cap + 1e-10), 0, 1)
            ax.imshow(v_display)
            if col == 0:
                ax.set_ylabel(model, fontsize=10, rotation=0, ha='right', va='center')
            if row == 0:
                ax.set_title(f'SV-{col+1}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_grid.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
