"""
aggregate_spectra.py  — Step 2b

Loads a directory of per-image .npy spectra and computes population-level
aggregates:
    S_mean  = mean spectrum across N images
    S_var   = variance spectrum across N images
    delta_S = S_mean_target - S_mean_reference   (requires --ref_dir)

Usage:
    # Aggregate each model's spectra:
    python scripts/aggregate_spectra.py \\
        --spec_dir results/phase1_sanity/spectra/base \\
        --out_dir  results/phase1_sanity/aggregates/base

    python scripts/aggregate_spectra.py \\
        --spec_dir results/phase1_sanity/spectra/clean \\
        --out_dir  results/phase1_sanity/aggregates/clean

    # For poisoned: also compute delta_S against base reference:
    python scripts/aggregate_spectra.py \\
        --spec_dir results/phase1_sanity/spectra/poisoned \\
        --ref_dir  results/phase1_sanity/aggregates/base \\
        --out_dir  results/phase1_sanity/aggregates/poisoned

    # Also compute delta_S for clean vs base (should be near-zero):
    python scripts/aggregate_spectra.py \\
        --spec_dir results/phase1_sanity/spectra/clean \\
        --ref_dir  results/phase1_sanity/aggregates/base \\
        --out_dir  results/phase1_sanity/aggregates/clean

CPU-only. Run on login node.
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_spectra(spec_dir: Path, max_n: int = None) -> np.ndarray:
    """Load all .npy files from spec_dir into a stacked array (N, H, W)."""
    paths = sorted(spec_dir.glob('*.npy'))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {spec_dir}")
    if max_n is not None:
        paths = paths[:max_n]

    # Load first to get shape
    sample = np.load(paths[0])
    H, W = sample.shape

    spectra = np.empty((len(paths), H, W), dtype=np.float32)
    spectra[0] = sample

    for i, p in enumerate(tqdm(paths[1:], desc='  Loading spectra', unit='spec'), start=1):
        spectra[i] = np.load(p)

    return spectra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_dir', required=True,
                        help='Directory of per-image .npy spectra (from compute_spectra.py)')
    parser.add_argument('--out_dir',  required=True,
                        help='Output directory for aggregate .npy files')
    parser.add_argument('--ref_dir',  default=None,
                        help='Aggregate dir of reference model (for delta_S). '
                             'Pass the OUT_DIR of the base model run.')
    parser.add_argument('--max_n',    type=int, default=None,
                        help='Use only the first N spectra (for ablation studies)')
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating spectra from: {spec_dir}")
    if args.max_n:
        print(f"  Using first {args.max_n} spectra (ablation mode)")

    spectra = load_spectra(spec_dir, max_n=args.max_n)
    N, H, W = spectra.shape
    print(f"  Loaded {N} spectra, shape ({H}, {W})")

    # S_mean and S_var
    print("  Computing S_mean and S_var ...")
    S_mean = np.mean(spectra, axis=0).astype(np.float32)
    S_var  = np.var(spectra,  axis=0, ddof=1).astype(np.float32)

    np.save(out_dir / 'S_mean.npy', S_mean)
    np.save(out_dir / 'S_var.npy',  S_var)
    print(f"  S_mean — min: {S_mean.min():.4f}, max: {S_mean.max():.4f}")
    print(f"  S_var  — min: {S_var.min():.4f},  max: {S_var.max():.4f}")

    # delta_S = S_mean_target - S_mean_reference
    if args.ref_dir is not None:
        ref_dir = Path(args.ref_dir)
        ref_mean_path = ref_dir / 'S_mean.npy'
        if not ref_mean_path.exists():
            raise FileNotFoundError(
                f"Reference S_mean.npy not found at {ref_mean_path}. "
                "Run aggregate_spectra.py on the base model first."
            )
        S_mean_ref = np.load(ref_mean_path)
        delta_S = (S_mean - S_mean_ref).astype(np.float32)
        np.save(out_dir / 'delta_S.npy', delta_S)
        print(f"  delta_S — min: {delta_S.min():.4f}, max: {delta_S.max():.4f}, "
              f"abs_mean: {np.abs(delta_S).mean():.4f}")
    else:
        print("  (No --ref_dir supplied — skipping delta_S)")

    # Save metadata
    meta = {
        'n_spectra': int(N),
        'shape': [int(H), int(W)],
        'spec_dir': str(spec_dir),
        'ref_dir': str(args.ref_dir) if args.ref_dir else None,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nAggregates saved to: {out_dir}")
    print(f"  S_mean.npy, S_var.npy" + (", delta_S.npy" if args.ref_dir else ""))


if __name__ == '__main__':
    main()
