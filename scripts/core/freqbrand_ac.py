"""
freqbrand_ac.py — FreqBrand-AC: Autocorrelation Split-Half Consistency Test

Detects hidden consistent artifacts in generated image populations by computing
per-image autocorrelation features and testing split-half consistency via
cosine similarity of mean feature vectors.

Operates on RAW PNGs (not BM3D residuals). Bypasses denoiser entirely.
Translation-invariant, frequency-neutral, reference-free.

Based on Sina Mansouri's specification (split half autocorr experiment.pdf).

Usage:
    python scripts/freqbrand_ac.py \
        --image_dir results/phase1_populations/logo_hf \
        --model_name logo_hf \
        --output_dir results/phase2_5/ac/logo_hf \
        --max_lag 32 \
        --n_splits 100 \
        --seed 42
"""

import argparse
import json
import os
import numpy as np
from numpy.fft import fft2, ifft2
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def autocorr_features(img, max_lag=32):
    """Extract autocorrelation features from a single image.

    Per Sina's spec:
    1. Convert to grayscale float64, z-normalize
    2. Zero-padded FFT (prevents circular wraparound)
    3. Autocorrelation via Wiener-Khinchin: real(ifft2(F * conj(F)))
    4. Extract center region (lags -max_lag to +max_lag)
    5. Normalize by max value
    6. Flatten to (2*max_lag+1)^2 dimensional vector

    Args:
        img: numpy array, shape (H, W, 3), uint8
        max_lag: maximum lag in each direction (default 32 → 65x65 = 4225 dims)

    Returns:
        1D numpy array of shape ((2*max_lag+1)**2,)
    """
    gray = np.mean(img, axis=2).astype(np.float64)
    gray = (gray - gray.mean()) / (gray.std() + 1e-8)

    # Zero-padded FFT — mandatory to avoid circular wraparound
    F = fft2(gray, s=[gray.shape[0] * 2, gray.shape[1] * 2])
    ac = np.real(ifft2(F * np.conj(F)))

    # Roll to center the zero-lag, then extract ±max_lag region
    ac = np.roll(ac, shift=max_lag, axis=0)
    ac = np.roll(ac, shift=max_lag, axis=1)
    patch = ac[:2 * max_lag + 1, :2 * max_lag + 1]

    # Normalize by max
    patch = patch / (patch.max() + 1e-10)
    return patch.flatten()


def split_half_cosine(features, n_splits=100, rng=None):
    """Split-half consistency score via cosine similarity of means.

    Per Sina's spec: split population into two random halves, compute
    mean feature vector of each half, take cosine similarity. Repeat
    n_splits times, return all scores.

    Args:
        features: numpy array, shape (N, D)
        n_splits: number of random splits
        rng: numpy random generator

    Returns:
        list of n_splits cosine similarity scores
    """
    if rng is None:
        rng = np.random.default_rng(42)

    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(len(features))
        half = len(idx) // 2
        A = np.mean(features[idx[:half]], axis=0)
        B = np.mean(features[idx[half:]], axis=0)
        cos = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-10)
        scores.append(float(cos))
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
    parser = argparse.ArgumentParser(description="FreqBrand-AC: Autocorrelation split-half")
    parser.add_argument("--image_dir", required=True, help="Directory of generated PNGs")
    parser.add_argument("--model_name", required=True, help="Name for this model/population")
    parser.add_argument("--output_dir", required=True, help="Where to save results")
    parser.add_argument("--max_lag", type=int, default=32, help="Max autocorrelation lag")
    parser.add_argument("--n_splits", type=int, default=100, help="Number of split-half iterations")
    parser.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FreqBrand-AC: Autocorrelation Split-Half")
    print(f"  Model:      {args.model_name}")
    print(f"  Image dir:  {args.image_dir}")
    print(f"  Max lag:    {args.max_lag} (feature dim: {(2*args.max_lag+1)**2})")
    print(f"  N splits:   {args.n_splits}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # Load images
    images = load_images(args.image_dir, args.n_images)
    print(f"  Loaded {len(images)} images")

    # Extract autocorrelation features
    print("  Extracting autocorrelation features...")
    features = np.array([
        autocorr_features(img, max_lag=args.max_lag)
        for img in tqdm(images, desc="AC features", unit="img")
    ])
    print(f"  Feature matrix shape: {features.shape}")

    # Split-half consistency (cosine of means)
    print("  Running split-half consistency test...")
    scores = split_half_cosine(features, n_splits=args.n_splits, rng=rng)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    print(f"\n  Result: {args.model_name}")
    print(f"    Split-half cosine (mean ± std): {mean_score:.6f} ± {std_score:.6f}")
    print(f"    Min: {min(scores):.6f}, Max: {max(scores):.6f}")

    # Save results
    results = {
        "method": "freqbrand_ac",
        "model_name": args.model_name,
        "image_dir": str(args.image_dir),
        "n_images": len(images),
        "max_lag": args.max_lag,
        "feature_dim": (2 * args.max_lag + 1) ** 2,
        "n_splits": args.n_splits,
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

    # Also save features for potential reuse
    features_path = out_dir / "features.npy"
    np.save(features_path, features)
    print(f"  Features saved to {features_path}")


if __name__ == "__main__":
    main()
