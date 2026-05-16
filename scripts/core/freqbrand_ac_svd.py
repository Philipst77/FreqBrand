"""
freqbrand_ac_svd.py — FreqBrand-AC-SVD: AC features + SVD statistic (matched ablation)

Uses autocorrelation feature extraction (same as freqbrand_ac.py) but applies
PS's SVD-based split-half statistic instead of cosine-of-means. This isolates
the effect of AC's lag-truncation from the statistic choice, making any
AC-vs-PS difference interpretable.

Operates on RAW PNGs (not BM3D residuals).

Usage:
    python scripts/freqbrand_ac_svd.py \
        --image_dir results/phase1_populations/logo_hf \
        --model_name logo_hf \
        --output_dir results/phase2_5/ac_svd/logo_hf \
        --max_lag 32 \
        --n_splits 100 \
        --seed 42
"""

import argparse
import json
import numpy as np
from numpy.fft import fft2, ifft2
from pathlib import Path
from PIL import Image
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


def autocorr_features(img, max_lag=32):
    """Extract autocorrelation features from a single image.

    Identical to freqbrand_ac.py — zero-padded FFT autocorrelation,
    center ±max_lag region, normalize by max, flatten.
    """
    gray = np.mean(img, axis=2).astype(np.float64)
    gray = (gray - gray.mean()) / (gray.std() + 1e-8)

    F = fft2(gray, s=[gray.shape[0] * 2, gray.shape[1] * 2])
    ac = np.real(ifft2(F * np.conj(F)))

    ac = np.roll(ac, shift=max_lag, axis=0)
    ac = np.roll(ac, shift=max_lag, axis=1)
    patch = ac[:2 * max_lag + 1, :2 * max_lag + 1]

    patch = patch / (patch.max() + 1e-10)
    return patch.flatten()


def split_half_svd_cosine(features, n_splits=100, n_components=5, rng=None):
    """Split-half consistency using SVD statistic (matching PS's approach).

    1. Mean-center features (column-wise)
    2. Split into two halves
    3. randomized_svd on each half
    4. abs(dot(Vt_A[0], Vt_B[0]))
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Mean-center columns (matching PS)
    features_centered = features - features.mean(axis=0)

    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(len(features_centered))
        half = len(idx) // 2

        A = features_centered[idx[:half]]
        B = features_centered[idx[half:]]

        _, _, Vt_A = randomized_svd(A, n_components=n_components, random_state=None)
        _, _, Vt_B = randomized_svd(B, n_components=n_components, random_state=None)

        cos = float(np.abs(np.dot(Vt_A[0], Vt_B[0])))
        scores.append(cos)

    return scores


def load_images(image_dir, n_images=None):
    """Load PNG/JPG images from directory."""
    image_dir = Path(image_dir)
    paths = sorted(
        [p for p in image_dir.iterdir()
         if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    )
    if n_images is not None:
        paths = paths[:n_images]

    images = []
    for p in tqdm(paths, desc="Loading images", unit="img"):
        img = np.array(Image.open(p).convert("RGB"))
        images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser(description="FreqBrand-AC-SVD: AC features + SVD statistic")
    parser.add_argument("--image_dir", required=True, help="Directory of generated PNGs")
    parser.add_argument("--model_name", required=True, help="Name for this model/population")
    parser.add_argument("--output_dir", required=True, help="Where to save results")
    parser.add_argument("--max_lag", type=int, default=32, help="Max autocorrelation lag")
    parser.add_argument("--n_splits", type=int, default=100, help="Number of split-half iterations")
    parser.add_argument("--n_components", type=int, default=5, help="SVD components to compute")
    parser.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FreqBrand-AC-SVD: Autocorrelation + SVD Statistic (Ablation)")
    print(f"  Model:      {args.model_name}")
    print(f"  Image dir:  {args.image_dir}")
    print(f"  Max lag:    {args.max_lag} (feature dim: {(2*args.max_lag+1)**2})")
    print(f"  N splits:   {args.n_splits}")
    print(f"  SVD comps:  {args.n_components}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # Load images
    images = load_images(args.image_dir, args.n_images)
    print(f"  Loaded {len(images)} images")

    # Extract autocorrelation features (same as freqbrand_ac.py)
    print("  Extracting autocorrelation features...")
    features = np.array([
        autocorr_features(img, max_lag=args.max_lag)
        for img in tqdm(images, desc="AC features", unit="img")
    ])
    print(f"  Feature matrix shape: {features.shape}")

    # Split-half consistency (SVD cosine — matching PS's statistic)
    print("  Running split-half SVD consistency test...")
    scores = split_half_svd_cosine(
        features, n_splits=args.n_splits, n_components=args.n_components, rng=rng
    )

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    print(f"\n  Result: {args.model_name}")
    print(f"    Split-half SVD cosine (mean ± std): {mean_score:.6f} ± {std_score:.6f}")
    print(f"    Min: {min(scores):.6f}, Max: {max(scores):.6f}")

    # Save results
    results = {
        "method": "freqbrand_ac_svd",
        "model_name": args.model_name,
        "image_dir": str(args.image_dir),
        "n_images": len(images),
        "max_lag": args.max_lag,
        "feature_dim": (2 * args.max_lag + 1) ** 2,
        "n_splits": args.n_splits,
        "n_components": args.n_components,
        "seed": args.seed,
        "split_half_cosine_mean": mean_score,
        "split_half_cosine_std": std_score,
        "split_half_cosine_min": float(min(scores)),
        "split_half_cosine_max": float(max(scores)),
        "per_split_scores": scores,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {results_path}")


if __name__ == "__main__":
    main()
