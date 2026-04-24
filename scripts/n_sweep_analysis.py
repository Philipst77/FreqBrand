"""
n_sweep_analysis.py — Pre-registered N-sweep: detection vs sample size

Uses existing 500 residuals per model. Runs SVD at N={25, 50, 100, 250, 500}.
Reports σ₁/σ₂ ratio and leave-one-out detection at each N.

Pre-registered hypothesis (configs/n_sweep_hypothesis.md):
  - AUROC > 0.7 by N=100
  - AUROC > 0.95 by N=1000
  - Falsification: AUROC < 0.6 at N=500

Usage:
    python scripts/n_sweep_analysis.py                  # default 64x64
    python scripts/n_sweep_analysis.py --patch_size 128 # 128x128 primary
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/tmp/matplotlib'

import argparse
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
N_VALUES = [25, 50, 100, 250, 500]


def load_and_extract(residual_dir, patch_size, n_images):
    files = sorted(Path(residual_dir).glob("res_*.npy"))[:n_images]
    all_patches = []
    for f in files:
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


def leave_one_out(ratios):
    """For each model as suspect, check if its ratio > max(others)."""
    results = {}
    for suspect, s_ratio in ratios.items():
        max_other = max(v for k, v in ratios.items() if k != suspect)
        results[suspect] = s_ratio > max_other
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=64)
    args = parser.parse_args()

    PATCH_SIZE = args.patch_size
    patches_per_image = (1024 // PATCH_SIZE) ** 2

    ROOT = Path("/scratch/ygoonati/freqbrand")
    suffix = f"_ps{PATCH_SIZE}" if PATCH_SIZE != 64 else ""
    out_dir = ROOT / "results" / "phase1_diagnostics" / f"n_sweep{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, patches/image: {patches_per_image}")

    all_results = {}

    for N in N_VALUES:
        print(f"\n{'='*60}")
        print(f"N = {N} images (N_eff = {N * patches_per_image} patches)")
        print(f"{'='*60}")

        ratios = {}
        for model in MODELS:
            res_dir = str(ROOT / "results" / "phase1_residuals" / model)
            X = load_and_extract(res_dir, PATCH_SIZE, N)

            n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
            # Always use CPU randomized_svd with seed=42 for reproducibility
            # (matches svd_patch_analysis.py primary SVD path)
            _, S, _ = randomized_svd(X, n_components=n_comp, random_state=42)
            # True singular value ratio (not eigenvalue ratio)
            ratio = float(S[0] / S[1])
            ratios[model] = ratio
            print(f"  {model:25s} σ₁/σ₂={ratio:.4f}")

        # Leave-one-out
        loo = leave_one_out(ratios)
        poisoned_detected = loo['poisoned_avengers']
        false_positives = sum(1 for k, v in loo.items()
                              if v and k != 'poisoned_avengers')

        # Effect size
        clean_ratios = [ratios[m] for m in MODELS if m.startswith('clean_')]
        mean_clean = np.mean(clean_ratios)
        std_clean = np.std(clean_ratios, ddof=1) if len(clean_ratios) > 1 else 1e-10
        z = (ratios['poisoned_avengers'] - mean_clean) / std_clean
        gap = ratios['poisoned_avengers'] - max(clean_ratios)

        all_results[N] = {
            'ratios': ratios,
            'poisoned_detected': poisoned_detected,
            'false_positives': false_positives,
            'effect_size_z': float(z),
            'gap_over_max_clean': float(gap),
            'poisoned_ratio': ratios['poisoned_avengers'],
            'max_clean_ratio': max(clean_ratios),
        }

        print(f"  --- Detection: poisoned={'DETECTED' if poisoned_detected else 'MISSED'}, "
              f"FP={false_positives}, z={z:.1f}, gap={gap:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("N-SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':>6s} {'N_eff':>8s} {'Poisoned':>10s} {'MaxClean':>10s} "
          f"{'Gap':>8s} {'z':>6s} {'Det?':>5s} {'FP':>4s}")
    print("-" * 65)
    for N in N_VALUES:
        r = all_results[N]
        print(f"{N:6d} {N*patches_per_image:8d} {r['poisoned_ratio']:10.4f} {r['max_clean_ratio']:10.4f} "
              f"{r['gap_over_max_clean']:8.4f} {r['effect_size_z']:6.1f} "
              f"{'YES' if r['poisoned_detected'] else 'NO':>5s} {r['false_positives']:4d}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ns = N_VALUES
    poisoned_ratios = [all_results[n]['poisoned_ratio'] for n in ns]
    max_clean_ratios = [all_results[n]['max_clean_ratio'] for n in ns]
    zs = [all_results[n]['effect_size_z'] for n in ns]

    ax1.plot(ns, poisoned_ratios, 'ro-', label='Poisoned', linewidth=2)
    ax1.plot(ns, max_clean_ratios, 'bs-', label='Max clean', linewidth=2)
    ax1.fill_between(ns, max_clean_ratios, poisoned_ratios, alpha=0.2, color='red')
    ax1.set_xlabel('N (images)')
    ax1.set_ylabel('σ₁/σ₂ ratio')
    ax1.set_title(f'Detection Separation vs Sample Size ({PATCH_SIZE}x{PATCH_SIZE})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    ax2.plot(ns, zs, 'go-', linewidth=2)
    ax2.axhline(1.96, color='r', linestyle='--', alpha=0.5, label='z=1.96 (p<0.05)')
    ax2.set_xlabel('N (images)')
    ax2.set_ylabel('Effect size (z)')
    ax2.set_title(f'Effect Size vs Sample Size ({PATCH_SIZE}x{PATCH_SIZE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(out_dir / "n_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()

    with open(out_dir / "n_sweep_results.json", 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
