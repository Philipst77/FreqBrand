"""
statistical_detection.py — Method 2: Training-free per-frequency hypothesis testing

For each DCT frequency bin, runs a Welch two-sample t-test between the suspect
model's population of spectra and the base SDXL reference population.
Applies Benjamini-Hochberg FDR correction, then measures SPATIAL CLUSTERING
of the resulting significance map.

Key insight:
  - Poisoned model: significant bins form a tight, structured cluster in
    low-mid frequencies (the logo's spectral footprint).
  - Clean/legitimate model: significant bins are sparse or diffuse
    (generic style shift from finetuning).
  - Requires ZERO training on any poisoned model — purely statistical.

Auto-detects all model directories in spec_root (except base) and tests each
one vs base. New models (hf_logo_poisoned, clean_200, juggernaut) are picked
up automatically when their spectra directories exist.

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
        arr = np.load(p)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target, target), mode='area')
        pool[i] = t.squeeze().numpy()
    return pool


# ---------------------------------------------------------------------------
# Statistical test + FDR
# ---------------------------------------------------------------------------

def significance_map(pool_a: np.ndarray, pool_b: np.ndarray,
                     fdr_alpha: float = 0.05) -> tuple:
    """
    Welch t-test at each frequency bin (pool_a vs pool_b).
    Returns:
      p_raw      : (H, W) raw p-values
      significant: (H, W) bool, BH-FDR corrected significance map
    """
    _, p_raw = stats.ttest_ind(pool_a, pool_b, axis=0, equal_var=False)
    p_raw = np.nan_to_num(p_raw, nan=1.0)

    p_flat = p_raw.flatten()
    p_corrected = stats.false_discovery_control(p_flat, method='bh')
    significant = (p_corrected < fdr_alpha).reshape(p_raw.shape)

    return p_raw, significant


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def clustering_metrics(sig_map: np.ndarray) -> dict:
    """
    Quantify spatial clustering of the significance map.
    Returns n_significant, lcc_size, lcc_fraction, connected_fraction, low_mid_ratio.
    """
    H, W = sig_map.shape
    n_sig = int(sig_map.sum())
    n_total = H * W

    # Largest connected component
    labeled, n_components = scipy_label(sig_map)
    lcc_size = 0
    if n_components > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]
        lcc_size = int(component_sizes.max())

    # Connected fraction
    if n_sig > 0:
        p = np.pad(sig_map.astype(np.float32), 1)
        has_neighbor = (
            p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        ) > 0
        connected_fraction = float((sig_map & has_neighbor).sum()) / n_sig
    else:
        connected_fraction = 0.0

    # Low+mid vs high frequency ratio (DCT origin at [0,0])
    u = np.arange(H)[:, None]
    v = np.arange(W)[None, :]
    r = np.sqrt(u**2 + v**2)
    low_mid_mask = r < (H // 2)
    low_mid_ratio = float((sig_map & low_mid_mask).sum()) / max(n_sig, 1)

    return {
        'n_significant':      n_sig,
        'n_sig_fraction':     round(float(n_sig) / n_total, 4),
        'lcc_size':           lcc_size,
        'lcc_fraction':       round(float(lcc_size) / max(n_sig, 1), 4),
        'connected_fraction': round(connected_fraction, 4),
        'low_mid_ratio':      round(low_mid_ratio, 4),
    }


def detection_verdict(metrics: dict) -> str:
    """Classify a model based on clustering metrics."""
    if metrics['lcc_size'] > 500 and metrics['low_mid_ratio'] > 0.6:
        return 'POISONED'
    elif metrics['lcc_size'] > 200 or metrics['connected_fraction'] > 0.7:
        return 'SUSPICIOUS'
    else:
        return 'CLEAN'


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_all_significance_maps(results: dict, out_path: Path) -> None:
    """Grid of significance maps — one per model."""
    models = list(results.keys())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, model_name in enumerate(models):
        sig_map = results[model_name]['sig_map']
        p_raw   = results[model_name]['p_raw']
        metrics = results[model_name]['metrics']
        verdict = results[model_name]['verdict']

        # Row 0: significance map
        ax = axes[0, col]
        ax.imshow(sig_map, cmap='Reds', interpolation='nearest', aspect='auto')
        color = 'red' if verdict == 'POISONED' else ('orange' if verdict == 'SUSPICIOUS' else 'green')
        ax.set_title(
            f'{model_name}\n'
            f'[{verdict}]\n'
            f'N_sig={metrics["n_significant"]} ({metrics["n_sig_fraction"]*100:.1f}%)  '
            f'LCC={metrics["lcc_size"]}\n'
            f'LowMid={metrics["low_mid_ratio"]*100:.1f}%  '
            f'Conn={metrics["connected_fraction"]*100:.1f}%',
            fontsize=8, color=color
        )
        ax.set_xlabel('v (freq bin)')
        ax.set_ylabel('u (freq bin)')

        # Row 1: -log10 p-value map
        ax = axes[1, col]
        log_p = -np.log10(np.clip(p_raw, 1e-30, 1.0))
        im = ax.imshow(log_p, cmap='hot', interpolation='nearest', aspect='auto',
                       vmin=0, vmax=np.percentile(log_p, 99))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='-log10(p)')
        ax.set_title(f'{model_name} — p-value map', fontsize=8)

    plt.suptitle('Statistical Detection: Per-Frequency Welch t-test + BH FDR\n'
                 'Each model vs Base SDXL (N=1K images)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Significance maps: {out_path}")


def plot_clustering_comparison(results: dict, out_path: Path) -> None:
    """Bar chart comparing clustering metrics across all models."""
    models  = list(results.keys())
    metrics_keys = ['n_sig_fraction', 'lcc_fraction', 'connected_fraction', 'low_mid_ratio']
    labels  = ['Sig Fraction', 'LCC Fraction', 'Connected Frac', 'LowMid Ratio']

    colors = []
    for m in models:
        v = results[m]['verdict']
        colors.append('crimson' if v == 'POISONED' else ('orange' if v == 'SUSPICIOUS' else 'steelblue'))

    x = np.arange(len(metrics_keys))
    w = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        offset = (i - len(models) / 2 + 0.5) * w
        vals = [results[model_name]['metrics'][k] for k in metrics_keys]
        bars = ax.bar(x + offset, vals, w * 0.9, label=model_name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Score (0–1)')
    ax.set_title('Clustering Metrics by Model\nHigh scores → structured frequency cluster → likely poisoned')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Clustering comparison: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',  required=True,
                        help='Dir with base/, clean/, poisoned/, juggernaut/, etc. subdirs')
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--downsample', type=int,   default=256)
    parser.add_argument('--fdr_alpha',  type=float, default=0.05)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load base pool once
    # -----------------------------------------------------------------------
    print("Loading base spectra pool ...")
    pool_base = load_and_downsample_pool(spec_root / 'base', args.downsample)
    H, W = pool_base.shape[1], pool_base.shape[2]
    print(f"  Base pool: {pool_base.shape}")

    # -----------------------------------------------------------------------
    # Auto-detect all model directories (everything except base)
    # -----------------------------------------------------------------------
    model_dirs = sorted([
        d for d in spec_root.iterdir()
        if d.is_dir() and d.name != 'base' and len(list(d.glob('*.npy'))) > 0
    ])
    if not model_dirs:
        raise RuntimeError(f"No model spectra found in {spec_root} (other than base)")

    print(f"\nModels to test: {[d.name for d in model_dirs]}")

    # -----------------------------------------------------------------------
    # Run statistical test for each model
    # -----------------------------------------------------------------------
    all_results = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n{'='*50}")
        print(f"Testing {model_name} vs base ...")

        pool = load_and_downsample_pool(model_dir, args.downsample)
        p_raw, sig = significance_map(pool, pool_base, args.fdr_alpha)
        metrics = clustering_metrics(sig)
        verdict = detection_verdict(metrics)
        gc.collect()

        print(f"  Verdict: {verdict}")
        print(f"  N_significant: {metrics['n_significant']} ({metrics['n_sig_fraction']*100:.1f}%)")
        print(f"  LCC size: {metrics['lcc_size']}  fraction: {metrics['lcc_fraction']:.3f}")
        print(f"  Connected fraction: {metrics['connected_fraction']:.3f}")
        print(f"  Low+mid freq ratio: {metrics['low_mid_ratio']:.3f}")

        all_results[model_name] = {
            'sig_map': sig,
            'p_raw':   p_raw,
            'metrics': metrics,
            'verdict': verdict,
        }

        # Save per-model significance map
        np.save(out_dir / f'sig_map_{model_name}.npy', sig)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures ...")
    plot_all_significance_maps(all_results, out_dir / 'significance_maps_all.png')
    plot_clustering_comparison(all_results, out_dir / 'clustering_comparison.png')

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report = {
        'settings': {
            'downsample': args.downsample,
            'fdr_alpha':  args.fdr_alpha,
            'n_base':     int(pool_base.shape[0]),
        },
        'models': {
            name: {
                'n_spectra': int(pool_base.shape[0]),  # same N for all by default
                'metrics':   r['metrics'],
                'verdict':   r['verdict'],
            }
            for name, r in all_results.items()
        },
    }
    rp = out_dir / 'statistical_detection_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, r in all_results.items():
        v = r['verdict']
        m = r['metrics']
        print(f"  {name:25s}  [{v:10s}]  LCC={m['lcc_size']:5d}  "
              f"LowMid={m['low_mid_ratio']:.2f}  Conn={m['connected_fraction']:.2f}")
    print("\nStatistical detection complete.")


if __name__ == '__main__':
    main()
