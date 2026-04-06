"""
statistical_detection.py — Method 2: Training-free per-frequency hypothesis testing

For each DCT frequency bin independently, runs a Welch two-sample t-test between
the suspect model's population of spectra and the base SDXL reference population.
Applies Benjamini-Hochberg FDR correction, then measures the SPATIAL CLUSTERING
of significant bins in the resulting significance map.

Key insight:
  - Poisoned model: significant bins cluster tightly in a structured low-mid freq
    region (the logo's spectral footprint).
  - Clean LoRA: significant bins are sparse or diffuse (generic style shift).
  - This method requires ZERO training on any poisoned model — purely statistical.

Runs both comparisons:
  1. Poisoned LoRA vs Base SDXL
  2. Clean LoRA   vs Base SDXL

Usage:
    python scripts/statistical_detection.py \
        --spec_root results/phase3_spectra/spectra \
        --out_dir   results/phase3_statistical \
        --downsample 256 \
        --fdr_alpha  0.05
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from scipy.ndimage import label as scipy_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)


# ---------------------------------------------------------------------------
# Data loading + downsampling
# ---------------------------------------------------------------------------

def load_and_downsample_pool(spec_dir: Path, target: int) -> np.ndarray:
    """Load all .npy spectra, downsample each to (target, target), return (N, H, W)."""
    paths = sorted(spec_dir.glob('*.npy'))
    if not paths:
        raise FileNotFoundError(f"No spectra in {spec_dir}")
    print(f"  Loading {len(paths)} spectra from {spec_dir.name} → {target}×{target} ...")
    pool = np.empty((len(paths), target, target), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, leave=False)):
        arr = np.load(p)   # (H, W) float32
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        t = F.interpolate(t, size=(target, target), mode='area')
        pool[i] = t.squeeze().numpy()
    return pool


# ---------------------------------------------------------------------------
# Statistical test + FDR
# ---------------------------------------------------------------------------

def significance_map(pool_a: np.ndarray, pool_b: np.ndarray,
                     fdr_alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch t-test at each frequency bin (pool_a vs pool_b).
    Returns:
      p_raw      : (H, W) raw p-values
      significant: (H, W) bool, BH-FDR corrected significance map
    """
    _, p_raw = stats.ttest_ind(pool_a, pool_b, axis=0, equal_var=False)
    p_raw = np.nan_to_num(p_raw, nan=1.0)

    # Benjamini-Hochberg FDR correction (scipy 1.11+)
    p_flat = p_raw.flatten()
    p_corrected = stats.false_discovery_control(p_flat, method='bh')
    significant = (p_corrected < fdr_alpha).reshape(p_raw.shape)

    return p_raw, significant


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def clustering_metrics(sig_map: np.ndarray, H: int, W: int) -> dict:
    """
    Quantify spatial clustering of the significance map.

    Returns:
      n_significant      : total significant bins
      n_sig_fraction     : fraction of all bins that are significant
      lcc_size           : largest connected component (4-connectivity)
      lcc_fraction       : LCC size as fraction of all significant bins
      connected_fraction : fraction of significant bins with ≥1 significant neighbor
      low_mid_ratio      : fraction of significant bins in low+mid freq (r < H//2)
    """
    n_sig = int(sig_map.sum())
    n_total = H * W

    # Largest connected component
    labeled, n_components = scipy_label(sig_map)
    lcc_size = 0
    if n_components > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]   # skip background (0)
        lcc_size = int(component_sizes.max())

    # Connected fraction: significant bins with ≥1 significant neighbor
    if n_sig > 0:
        # Pad and shift to check 4-neighbors
        p = np.pad(sig_map.astype(np.float32), 1)
        has_neighbor = (
            p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        ) > 0
        connected = (sig_map & has_neighbor).sum()
        connected_fraction = float(connected) / n_sig
    else:
        connected_fraction = 0.0

    # Band ratio: significant bins in low+mid vs high freq
    # For DCT at origin [0,0], radius = sqrt(u^2 + v^2)
    u = np.arange(H)[:, None]
    v = np.arange(W)[None, :]
    r = np.sqrt(u**2 + v**2)
    threshold = H // 2   # radial cutoff between mid and high freq
    low_mid_mask = r < threshold
    sig_low_mid = int((sig_map & low_mid_mask).sum())
    low_mid_ratio = float(sig_low_mid) / max(n_sig, 1)

    return {
        'n_significant':      n_sig,
        'n_sig_fraction':     round(float(n_sig) / n_total, 4),
        'lcc_size':           lcc_size,
        'lcc_fraction':       round(float(lcc_size) / max(n_sig, 1), 4),
        'connected_fraction': round(connected_fraction, 4),
        'low_mid_ratio':      round(low_mid_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_significance_maps(sig_map_poisoned: np.ndarray, sig_map_clean: np.ndarray,
                           p_raw_poisoned: np.ndarray, p_raw_clean: np.ndarray,
                           metrics_poisoned: dict, metrics_clean: dict,
                           out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    def _plot_sig(ax, sig_map, metrics, title):
        ax.imshow(sig_map, cmap='Reds', interpolation='nearest', aspect='auto')
        ax.set_title(f'{title}\n'
                     f'N_sig={metrics["n_significant"]} ({metrics["n_sig_fraction"]*100:.1f}%)  '
                     f'LCC={metrics["lcc_size"]}  '
                     f'LowMid={metrics["low_mid_ratio"]*100:.1f}%',
                     fontsize=9)
        ax.set_xlabel('Frequency bin v')
        ax.set_ylabel('Frequency bin u')

    def _plot_pval(ax, p_raw, title):
        # -log10 p-value map (clipped for display)
        log_p = -np.log10(np.clip(p_raw, 1e-30, 1.0))
        im = ax.imshow(log_p, cmap='hot', interpolation='nearest', aspect='auto',
                       vmin=0, vmax=np.percentile(log_p, 99))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='-log10(p)')
        ax.set_title(f'{title}\n(-log10 p-value)', fontsize=9)

    _plot_sig(axes[0, 0], sig_map_poisoned, metrics_poisoned,
              'Significance Map — Poisoned LoRA vs Base SDXL')
    _plot_sig(axes[1, 0], sig_map_clean,    metrics_clean,
              'Significance Map — Clean LoRA vs Base SDXL')

    _plot_pval(axes[0, 1], p_raw_poisoned, 'P-value map — Poisoned LoRA')
    _plot_pval(axes[1, 1], p_raw_clean,    'P-value map — Clean LoRA')

    # Clustering score bar chart
    ax = axes[0, 2]
    metrics_keys = ['n_sig_fraction', 'lcc_fraction', 'connected_fraction', 'low_mid_ratio']
    labels = ['Sig\nFraction', 'LCC\nFraction', 'Connected\nFraction', 'LowMid\nRatio']
    x = np.arange(len(metrics_keys))
    w = 0.35
    ax.bar(x - w/2, [metrics_poisoned[k] for k in metrics_keys],
           w, label='Poisoned LoRA', color='crimson', alpha=0.85)
    ax.bar(x + w/2, [metrics_clean[k] for k in metrics_keys],
           w, label='Clean LoRA', color='steelblue', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score (higher = more clustered)')
    ax.set_title('Clustering Metrics Comparison')
    ax.legend()
    ax.set_ylim(0, 1)

    # Difference map (significance map: poisoned - clean)
    ax = axes[1, 2]
    diff = sig_map_poisoned.astype(float) - sig_map_clean.astype(float)
    im = ax.imshow(diff, cmap='RdBu_r', interpolation='nearest', aspect='auto',
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Poisoned − Clean')
    ax.set_title('Significance Map Difference\n(red = only in poisoned, blue = only in clean)',
                 fontsize=9)

    plt.suptitle('Statistical Detection: Per-Frequency Bin Significance (Welch t-test + BH FDR)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True)
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--downsample',  type=int, default=256,
                        help='Downsample spectra to this resolution before testing')
    parser.add_argument('--fdr_alpha',   type=float, default=0.05)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load pools
    # -----------------------------------------------------------------------
    print("Loading and downsampling spectra pools ...")
    pool_base     = load_and_downsample_pool(spec_root / 'base',     args.downsample)
    pool_poisoned = load_and_downsample_pool(spec_root / 'poisoned', args.downsample)
    pool_clean    = load_and_downsample_pool(spec_root / 'clean',    args.downsample)
    H, W = pool_base.shape[1], pool_base.shape[2]
    print(f"  Pool shapes: base={pool_base.shape}, poisoned={pool_poisoned.shape}, "
          f"clean={pool_clean.shape}")

    # -----------------------------------------------------------------------
    # Statistical tests
    # -----------------------------------------------------------------------
    print(f"\nRunning Welch t-test + BH FDR (alpha={args.fdr_alpha}) ...")
    print("  Poisoned vs Base ...")
    p_raw_poisoned, sig_poisoned = significance_map(pool_poisoned, pool_base, args.fdr_alpha)
    gc.collect()

    print("  Clean LoRA vs Base ...")
    p_raw_clean, sig_clean = significance_map(pool_clean, pool_base, args.fdr_alpha)
    gc.collect()

    # -----------------------------------------------------------------------
    # Clustering metrics
    # -----------------------------------------------------------------------
    print("\nComputing clustering metrics ...")
    metrics_poisoned = clustering_metrics(sig_poisoned, H, W)
    metrics_clean    = clustering_metrics(sig_clean,    H, W)

    print("\nPoisoned LoRA vs Base:")
    for k, v in metrics_poisoned.items():
        print(f"  {k}: {v}")
    print("\nClean LoRA vs Base:")
    for k, v in metrics_clean.items():
        print(f"  {k}: {v}")

    # Detection verdict based on clustering score gap
    # A clear poisoning signal: poisoned LCC > 5× clean LCC and low_mid_ratio > 0.7
    lcc_ratio = metrics_poisoned['lcc_size'] / max(metrics_clean['lcc_size'], 1)
    print(f"\nLCC ratio (poisoned / clean): {lcc_ratio:.2f}x")
    if lcc_ratio > 3.0 and metrics_poisoned['low_mid_ratio'] > 0.6:
        verdict = "POISONING DETECTED — significant clustering in low-mid frequency band"
    elif lcc_ratio > 1.5:
        verdict = "WEAK SIGNAL — moderate clustering difference, inconclusive"
    else:
        verdict = "NO SIGNAL — clustering similar between poisoned and clean"
    print(f"Verdict: {verdict}")

    # -----------------------------------------------------------------------
    # Save significance maps + report
    # -----------------------------------------------------------------------
    np.save(out_dir / 'significance_map_poisoned.npy', sig_poisoned)
    np.save(out_dir / 'significance_map_clean.npy',    sig_clean)

    report = {
        'settings': {
            'downsample': args.downsample,
            'fdr_alpha': args.fdr_alpha,
            'n_base': int(pool_base.shape[0]),
            'n_poisoned': int(pool_poisoned.shape[0]),
            'n_clean': int(pool_clean.shape[0]),
        },
        'poisoned_vs_base': metrics_poisoned,
        'clean_vs_base':    metrics_clean,
        'lcc_ratio_poisoned_over_clean': round(float(lcc_ratio), 4),
        'verdict': verdict,
    }
    with open(out_dir / 'statistical_detection_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_dir}/statistical_detection_report.json")

    # -----------------------------------------------------------------------
    # Visualize
    # -----------------------------------------------------------------------
    print("\nGenerating figures ...")
    plot_significance_maps(
        sig_poisoned, sig_clean, p_raw_poisoned, p_raw_clean,
        metrics_poisoned, metrics_clean,
        out_dir / 'significance_comparison.png'
    )

    print("\nStatistical detection complete.")


if __name__ == '__main__':
    main()
