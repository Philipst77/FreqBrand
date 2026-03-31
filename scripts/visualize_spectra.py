"""
visualize_spectra.py  — Step 2c

Produces publication-quality figures from aggregate spectra.

Figure 1 — spectral_overview.png  (main paper figure):
    5-panel row: S_mean_base | S_mean_clean | S_mean_poisoned
                 | delta_S (poisoned-base) | delta_S (clean-base)

Figure 2 — delta_S_comparison.png  (close-up, paper figure):
    Side-by-side: delta_S poisoned vs delta_S clean, same colorscale

Figure 3 — S_var_comparison.png:
    S_var for each model side by side

Notes on visualization choices:
- DC component (0,0) is clipped to the 95th percentile of the rest of the
  spectrum — it dominates the colorscale otherwise.
- S_mean/S_var use 'inferno' colormap (perceptually uniform, print-friendly).
- delta_S uses 'RdBu_r' diverging colormap centered at 0.
- Colorscales are percentile-clipped (1st-99th) to handle outliers.

Usage:
    python scripts/visualize_spectra.py \\
        --base_dir    results/phase1_sanity/aggregates/base \\
        --clean_dir   results/phase1_sanity/aggregates/clean \\
        --poisoned_dir results/phase1_sanity/aggregates/poisoned \\
        --out_dir     results/phase1_sanity/spectral_figures

CPU-only. Run on login node.
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def load_aggregate(agg_dir: Path, key: str) -> np.ndarray | None:
    """Load a .npy aggregate file, return None if missing."""
    path = agg_dir / f'{key}.npy'
    if not path.exists():
        return None
    return np.load(path)


def suppress_dc(arr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Replace the DC component (0,0) with the percentile value of the rest."""
    arr = arr.copy()
    rest = arr[1:, 1:].flatten()
    arr[0, 0] = np.percentile(rest, percentile)
    return arr


def pclip(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) from percentile clipping."""
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def add_panel(ax, data: np.ndarray, title: str, cmap: str,
              vmin=None, vmax=None, label: str = ''):
    data = suppress_dc(data)
    if vmin is None or vmax is None:
        vmin, vmax = pclip(data)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='upper', aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=8)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',     required=True)
    parser.add_argument('--clean_dir',    required=True)
    parser.add_argument('--poisoned_dir', required=True)
    parser.add_argument('--out_dir',      required=True)
    args = parser.parse_args()

    base_dir     = Path(args.base_dir)
    clean_dir    = Path(args.clean_dir)
    poisoned_dir = Path(args.poisoned_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all aggregates
    S_mean   = {k: load_aggregate(d, 'S_mean')
                for k, d in [('base', base_dir), ('clean', clean_dir), ('poisoned', poisoned_dir)]}
    S_var    = {k: load_aggregate(d, 'S_var')
                for k, d in [('base', base_dir), ('clean', clean_dir), ('poisoned', poisoned_dir)]}
    delta_S  = {k: load_aggregate(d, 'delta_S')
                for k, d in [('clean', clean_dir), ('poisoned', poisoned_dir)]}

    for k, v in S_mean.items():
        if v is None:
            raise FileNotFoundError(f"S_mean.npy missing for {k}. Run aggregate_spectra.py first.")

    # Read N from meta.json for titles
    def get_n(d):
        meta_path = d / 'meta.json'
        if meta_path.exists():
            return json.load(open(meta_path))['n_spectra']
        return '?'

    n = {k: get_n(d) for k, d in [('base', base_dir), ('clean', clean_dir), ('poisoned', poisoned_dir)]}

    # -----------------------------------------------------------------------
    # Figure 1: 5-panel overview
    # -----------------------------------------------------------------------
    print("Generating spectral_overview.png ...")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle('FreqBrand — Population-Level DCT Spectra', fontsize=13, fontweight='bold', y=1.02)

    # Shared vmin/vmax for S_mean panels (comparable colorscale)
    all_means = np.stack([suppress_dc(S_mean[k]) for k in ('base', 'clean', 'poisoned')])
    mean_vmin, mean_vmax = pclip(all_means)

    add_panel(axes[0], S_mean['base'],     f'S_mean — Base SDXL\n(N={n["base"]})',
              'inferno', mean_vmin, mean_vmax, 'log |DCT|')
    add_panel(axes[1], S_mean['clean'],    f'S_mean — Clean LoRA\n(N={n["clean"]})',
              'inferno', mean_vmin, mean_vmax, 'log |DCT|')
    add_panel(axes[2], S_mean['poisoned'], f'S_mean — Poisoned LoRA\n(N={n["poisoned"]})',
              'inferno', mean_vmin, mean_vmax, 'log |DCT|')

    # delta_S panels — shared symmetric colorscale
    if delta_S['poisoned'] is not None and delta_S['clean'] is not None:
        all_deltas = np.stack([suppress_dc(delta_S['poisoned']), suppress_dc(delta_S['clean'])])
        dv = max(abs(float(np.percentile(all_deltas, 1))),
                 abs(float(np.percentile(all_deltas, 99))))
        add_panel(axes[3], delta_S['poisoned'],
                  f'ΔS — Poisoned − Base\n(N={n["poisoned"]})',
                  'RdBu_r', -dv, dv, 'Δ log |DCT|')
        add_panel(axes[4], delta_S['clean'],
                  f'ΔS — Clean − Base\n(N={n["clean"]})',
                  'RdBu_r', -dv, dv, 'Δ log |DCT|')
    else:
        for ax in axes[3:]:
            ax.text(0.5, 0.5, 'delta_S\nnot available\n(run aggregate_spectra.py\nwith --ref_dir)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_dir / 'spectral_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'spectral_overview.png'}")

    # -----------------------------------------------------------------------
    # Figure 2: delta_S close-up (key paper figure)
    # -----------------------------------------------------------------------
    if delta_S['poisoned'] is not None and delta_S['clean'] is not None:
        print("Generating delta_S_comparison.png ...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle('ΔS = S_mean_model − S_mean_base\n'
                     'Structured residual = logo spectral fingerprint',
                     fontsize=12, fontweight='bold')

        dv = max(abs(float(np.percentile(suppress_dc(delta_S['poisoned']), 1))),
                 abs(float(np.percentile(suppress_dc(delta_S['poisoned']), 99))),
                 abs(float(np.percentile(suppress_dc(delta_S['clean']), 1))),
                 abs(float(np.percentile(suppress_dc(delta_S['clean']), 99))))

        add_panel(axes[0], delta_S['poisoned'],
                  f'Poisoned LoRA (N={n["poisoned"]})\n← structured residual expected',
                  'RdBu_r', -dv, dv, 'Δ log |DCT|')
        add_panel(axes[1], delta_S['clean'],
                  f'Clean LoRA (N={n["clean"]})\n← should be near-zero / unstructured',
                  'RdBu_r', -dv, dv, 'Δ log |DCT|')

        plt.tight_layout()
        fig.savefig(out_dir / 'delta_S_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / 'delta_S_comparison.png'}")

    # -----------------------------------------------------------------------
    # Figure 3: S_var comparison
    # -----------------------------------------------------------------------
    print("Generating S_var_comparison.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('S_var — Frequency Variance Across Population\n'
                 'Low-variance bands in poisoned model = consistently present artifact',
                 fontsize=11, fontweight='bold')

    all_vars = np.stack([suppress_dc(S_var[k]) for k in ('base', 'clean', 'poisoned')])
    var_vmin, var_vmax = pclip(all_vars)

    for ax, k, label in zip(axes, ('base', 'clean', 'poisoned'),
                             ('Base SDXL', 'Clean LoRA', 'Poisoned LoRA')):
        add_panel(ax, S_var[k], f'S_var — {label}\n(N={n[k]})',
                  'viridis', var_vmin, var_vmax, 'Var(log |DCT|)')

    plt.tight_layout()
    fig.savefig(out_dir / 'S_var_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'S_var_comparison.png'}")

    print(f"\nAll figures saved to: {out_dir}")
    print("Key figure for paper: delta_S_comparison.png")
    print("  → Does poisoned ΔS show structured pattern while clean ΔS is flat?")
    print("  → If yes at N=50: strong early result. If noisy: need more images.")


if __name__ == '__main__':
    main()
