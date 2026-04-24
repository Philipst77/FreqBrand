"""
anisotropy_detection.py — Method 3: Spectral anisotropy decomposition

Decomposes delta_S (= S_mean_model - S_mean_base) into:
  1. Isotropic component:  the part that is uniform across all angles at each radius
  2. Anisotropic residual: the part that depends on direction (angle), not just radius

Key insight:
  - Legitimate style finetuning shifts the radial power profile uniformly
    → low anisotropy ratio (isotropic shift)
  - Logo embedding creates directional structure at specific orientations
    → high anisotropy ratio (anisotropic residual dominates)

This directly answers the reviewer objection:
  "What if the model was finetuned on vintage photos / anime / any stylized data?"
  → A style shift is isotropic. A logo is not.

Runs on the pre-computed aggregate files (S_mean.npy, delta_S.npy).
No spectra loading needed — just the aggregate .npy files.

Usage:
    python scripts/anisotropy_detection.py \
        --agg_root results/phase3_spectra/aggregates \
        --out_dir  results/phase3_anisotropy
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)


# ---------------------------------------------------------------------------
# Polar decomposition
# ---------------------------------------------------------------------------

def polar_decompose(delta_S: np.ndarray, n_radial_bins: int = 512,
                    n_angle_bins: int = 90) -> dict:
    """
    Decompose delta_S (H, W) into isotropic + anisotropic components.

    DCT origin (DC component) is at [0, 0] (top-left corner).
    Polar coordinates: r = sqrt(u^2 + v^2), theta = atan2(v, u)

    Returns dict with:
      isotropic_map    : (H, W) — isotropic component image
      anisotropic_map  : (H, W) — anisotropic residual image
      radial_profile   : (n_radial_bins,) — mean delta_S at each radius
      angular_profile  : (n_angle_bins,) — mean anisotropic residual at each angle
      anisotropy_ratio : scalar — ||anisotropic||_F / ||delta_S||_F
      r_max            : max radius in the map
    """
    H, W = delta_S.shape

    # Coordinate grids (DCT origin at [0, 0])
    u = np.arange(H, dtype=np.float32)[:, None]
    v = np.arange(W, dtype=np.float32)[None, :]
    r     = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)   # [0, pi/2] since u,v >= 0

    r_max = float(r.max())
    r_bins = np.linspace(0, r_max, n_radial_bins + 1)

    # Radial profile: mean delta_S at each radial bin
    radial_profile = np.zeros(n_radial_bins, dtype=np.float32)
    isotropic_map  = np.zeros_like(delta_S)

    for b in range(n_radial_bins):
        mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
        if mask.any():
            mean_val = float(delta_S[mask].mean())
            radial_profile[b] = mean_val
            isotropic_map[mask] = mean_val

    # Anisotropic residual
    anisotropic_map = delta_S - isotropic_map

    # Anisotropy ratio: energy of residual / energy of delta_S
    norm_aniso  = float(np.linalg.norm(anisotropic_map, 'fro'))
    norm_total  = float(np.linalg.norm(delta_S, 'fro'))
    anisotropy_ratio = norm_aniso / max(norm_total, 1e-8)

    # Angular profile of the anisotropic residual
    # Bin by angle to find which directions have excess anisotropic energy
    theta_bins = np.linspace(0, np.pi / 2, n_angle_bins + 1)
    angular_profile = np.zeros(n_angle_bins, dtype=np.float32)
    for b in range(n_angle_bins):
        mask = (theta >= theta_bins[b]) & (theta < theta_bins[b + 1])
        if mask.any():
            angular_profile[b] = float(np.abs(anisotropic_map[mask]).mean())

    return {
        'isotropic_map':    isotropic_map,
        'anisotropic_map':  anisotropic_map,
        'radial_profile':   radial_profile,
        'angular_profile':  angular_profile,
        'anisotropy_ratio': anisotropy_ratio,
        'r_max':            r_max,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_decomposition(delta_S: np.ndarray, result: dict,
                       model_name: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmax = np.percentile(np.abs(delta_S), 98)

    def _imshow(ax, data, title, cmap='RdBu_r', vmin=None, vmax_=None):
        v = vmax_ if vmax_ is not None else vmax
        im = ax.imshow(data, cmap=cmap, vmin=-v, vmax=v,
                       interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Frequency bin v')
        ax.set_ylabel('Frequency bin u')

    _imshow(axes[0, 0], delta_S, f'delta_S = S_mean({model_name}) − S_mean(base)')
    _imshow(axes[0, 1], result['isotropic_map'],
            'Isotropic component\n(radial mean — style shift)')
    _imshow(axes[0, 2], result['anisotropic_map'],
            f'Anisotropic residual\n(anisotropy ratio = {result["anisotropy_ratio"]:.4f})',
            vmax_=np.percentile(np.abs(result['anisotropic_map']), 99))

    # Radial profile
    ax = axes[1, 0]
    n_bins = len(result['radial_profile'])
    r_centers = np.linspace(0, result['r_max'], n_bins)
    ax.plot(r_centers, result['radial_profile'], color='darkorange', linewidth=1.5)
    ax.set_xlabel('Radius (frequency units from DC)')
    ax.set_ylabel('Mean delta_S')
    ax.set_title('Radial Profile (isotropic component)')
    ax.axhline(0, color='k', linewidth=0.5)

    # Angular profile of anisotropic residual
    ax = axes[1, 1]
    n_abins = len(result['angular_profile'])
    angles_deg = np.linspace(0, 90, n_abins)
    ax.plot(angles_deg, result['angular_profile'], color='steelblue', linewidth=1.5)
    ax.set_xlabel('Angle θ (degrees from u-axis)')
    ax.set_ylabel('Mean |anisotropic residual|')
    ax.set_title('Angular Profile of Anisotropic Residual\n'
                 '(peaks = dominant logo edge orientations)')
    ax.set_xlim(0, 90)

    # Abs anisotropic map (easier to see structure)
    ax = axes[1, 2]
    im = ax.imshow(np.abs(result['anisotropic_map']), cmap='hot',
                   interpolation='nearest', aspect='auto',
                   vmax=np.percentile(np.abs(result['anisotropic_map']), 99))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('|Anisotropic residual| (directional energy)', fontsize=10)

    plt.suptitle(f'Spectral Anisotropy Decomposition — {model_name}\n'
                 f'Anisotropy ratio = {result["anisotropy_ratio"]:.4f}  '
                 f'(higher = more directional structure = more logo-like)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison(results: dict, out_path: Path) -> None:
    """Side-by-side anisotropy ratio bar chart for all models."""
    names  = list(results.keys())
    ratios = [results[n]['anisotropy_ratio'] for n in names]
    colors = ['crimson' if 'poisoned' in n else 'steelblue' for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars = ax.bar(names, ratios, color=colors, alpha=0.85, edgecolor='k', linewidth=0.8)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{ratio:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Anisotropy Ratio  (||residual||_F / ||delta_S||_F)')
    ax.set_title('Spectral Anisotropy Ratio by Model\n'
                 'High ratio = directional structure (logo).  '
                 'Low ratio = isotropic shift (style).')
    ax.set_ylim(0, min(1.0, max(ratios) * 1.25))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg_root',      required=True,
                        help='Dir containing base/, clean/, poisoned/ aggregate subdirs')
    parser.add_argument('--out_dir',       required=True)
    parser.add_argument('--n_radial_bins', type=int, default=512)
    parser.add_argument('--n_angle_bins',  type=int, default=90)
    args = parser.parse_args()

    agg_root = Path(args.agg_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load aggregates
    # -----------------------------------------------------------------------
    def load_delta_S(model_name: str) -> np.ndarray:
        path = agg_root / model_name / 'delta_S.npy'
        if not path.exists():
            raise FileNotFoundError(f"delta_S.npy not found for {model_name} at {path}")
        arr = np.load(path).astype(np.float32)
        print(f"  Loaded delta_S for {model_name}: {arr.shape}")
        return arr

    print("Loading aggregate delta_S maps ...")
    # Discover which models have aggregates
    models_to_run = {}
    for m in ['poisoned', 'clean']:
        p = agg_root / m / 'delta_S.npy'
        if p.exists():
            models_to_run[m] = np.load(p).astype(np.float32)
            print(f"  Loaded {m}: {models_to_run[m].shape}")
        else:
            print(f"  WARNING: {p} not found, skipping {m}")

    # Also check for any extra models (hf_logo, wild, etc.)
    for d in sorted(agg_root.iterdir()):
        if d.is_dir() and d.name not in ('base', 'poisoned', 'clean') \
                and (d / 'delta_S.npy').exists():
            models_to_run[d.name] = np.load(d / 'delta_S.npy').astype(np.float32)
            print(f"  Loaded {d.name}: {models_to_run[d.name].shape}")

    if not models_to_run:
        raise RuntimeError("No delta_S.npy files found. Run aggregate_spectra.py first.")

    # -----------------------------------------------------------------------
    # Polar decomposition for each model
    # -----------------------------------------------------------------------
    all_results = {}
    for model_name, delta_S in models_to_run.items():
        print(f"\nDecomposing {model_name} ...")
        result = polar_decompose(delta_S, args.n_radial_bins, args.n_angle_bins)
        all_results[model_name] = result
        print(f"  Anisotropy ratio: {result['anisotropy_ratio']:.6f}")

        plot_decomposition(delta_S, result, model_name,
                           out_dir / f'anisotropy_{model_name}.png')

    # -----------------------------------------------------------------------
    # Comparison plot + report
    # -----------------------------------------------------------------------
    print("\nGenerating comparison figure ...")
    plot_comparison(
        {k: v for k, v in all_results.items()},
        out_dir / 'anisotropy_comparison.png'
    )

    report = {
        model_name: {
            'anisotropy_ratio': round(float(r['anisotropy_ratio']), 6),
            'radial_profile_mean':  round(float(r['radial_profile'].mean()), 6),
            'radial_profile_std':   round(float(r['radial_profile'].std()),  6),
            'angular_profile_max_bin': int(r['angular_profile'].argmax()),
            'angular_profile_max_deg': round(
                float(r['angular_profile'].argmax()) / args.n_angle_bins * 90, 1
            ),
        }
        for model_name, r in all_results.items()
    }

    if 'poisoned' in report and 'clean' in report:
        ratio_gap = report['poisoned']['anisotropy_ratio'] - report['clean']['anisotropy_ratio']
        report['anisotropy_gap_poisoned_minus_clean'] = round(ratio_gap, 6)
        if ratio_gap > 0.02:
            report['verdict'] = 'ANISOTROPY DETECTED — poisoned model has significantly higher directional structure'
        elif ratio_gap > 0.005:
            report['verdict'] = 'WEAK SIGNAL — modest anisotropy gap'
        else:
            report['verdict'] = 'NO SIGNAL — anisotropy similar between models'
        print(f"\nAnisotropy gap (poisoned − clean): {ratio_gap:.6f}")
        print(f"Verdict: {report['verdict']}")

    with open(out_dir / 'anisotropy_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_dir}/anisotropy_report.json")
    print("\nAnisotropy detection complete.")


if __name__ == '__main__':
    main()
