"""
freqbrand_ps.py — FreqBrand-PS: Power Spectrum Split-Half SVD Consistency Test

Detects hidden consistent artifacts in generated image populations by computing
per-image power spectra, mean-centering to remove shared 1/f² structure, and
testing split-half consistency via cosine similarity of top SVD singular vectors.

Operates on RAW PNGs (not BM3D residuals). Bypasses denoiser entirely.
Translation-invariant, frequency-neutral, reference-free.

Based on Sina Mansouri's specification (power spectrum split half experiment.pdf).

Usage:
    python scripts/freqbrand_ps.py \
        --image_dir results/phase1_populations/logo_hf \
        --model_name logo_hf \
        --output_dir results/phase2_5/ps/logo_hf \
        --resize 256 \
        --n_splits 100 \
        --seed 42
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


def power_spectrum_features(img, resize=256):
    """Compute power spectrum features for a single image.

    Per Sina's spec:
    1. Convert to grayscale float64, z-normalize
    2. Resize to (resize x resize) for tractability
    3. 2D FFT → |F|² (power spectrum)
    4. Flatten

    Args:
        img: numpy array, shape (H, W, 3), uint8
        resize: resize dimension before FFT (default 256 → 65536 dims)

    Returns:
        1D numpy array of shape (resize*resize,)
    """
    # Resize first (before grayscale conversion for quality)
    pil_img = Image.fromarray(img).resize((resize, resize), Image.LANCZOS)
    img_resized = np.array(pil_img)

    gray = np.mean(img_resized, axis=2).astype(np.float64)
    gray = (gray - gray.mean()) / (gray.std() + 1e-8)

    F = np.fft.fft2(gray)
    P = np.abs(F) ** 2

    return P.flatten()


def split_half_svd_cosine(spectra, n_splits=100, n_components=5, rng=None):
    """Split-half consistency score via cosine of top SVD singular vectors.

    Per Sina's spec:
    1. Mean-center columns (removes shared 1/f² baseline — CRITICAL)
    2. Split population into two random halves
    3. randomized_svd on each half
    4. abs(dot(Vt_A[0], Vt_B[0])) — cosine of top right-singular vectors
       with abs() for sign invariance
    5. Repeat n_splits times

    Args:
        spectra: numpy array, shape (N, D) — raw power spectra (NOT yet centered)
        n_splits: number of random splits
        n_components: number of SVD components to compute
        rng: numpy random generator

    Returns:
        list of n_splits cosine similarity scores
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Mean-center columns — removes shared 1/f² natural image spectrum
    # Without this, SVD just finds "all natural images look similar"
    spectra_centered = spectra - spectra.mean(axis=0)

    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(len(spectra_centered))
        half = len(idx) // 2

        A = spectra_centered[idx[:half]]
        B = spectra_centered[idx[half:]]

        # Randomized SVD — much faster than full SVD for high-dim data
        _, _, Vt_A = randomized_svd(A, n_components=n_components, random_state=None)
        _, _, Vt_B = randomized_svd(B, n_components=n_components, random_state=None)

        # Cosine of top right-singular vectors with abs for sign invariance
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
    parser = argparse.ArgumentParser(description="FreqBrand-PS: Power spectrum split-half SVD")
    parser.add_argument("--image_dir", required=True, help="Directory of generated PNGs")
    parser.add_argument("--model_name", required=True, help="Name for this model/population")
    parser.add_argument("--output_dir", required=True, help="Where to save results")
    parser.add_argument("--resize", type=int, default=256, help="Resize dimension before FFT")
    parser.add_argument("--n_splits", type=int, default=100, help="Number of split-half iterations")
    parser.add_argument("--n_components", type=int, default=5, help="SVD components to compute")
    parser.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = args.resize * args.resize

    print("=" * 60)
    print(f"FreqBrand-PS: Power Spectrum Split-Half SVD")
    print(f"  Model:       {args.model_name}")
    print(f"  Image dir:   {args.image_dir}")
    print(f"  Resize:      {args.resize}x{args.resize} (feature dim: {feature_dim})")
    print(f"  N splits:    {args.n_splits}")
    print(f"  SVD comps:   {args.n_components}")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)

    # Load images
    images = load_images(args.image_dir, args.n_images)
    print(f"  Loaded {len(images)} images")

    # Compute power spectra
    print("  Computing power spectra...")
    spectra = np.array([
        power_spectrum_features(img, resize=args.resize)
        for img in tqdm(images, desc="Power spectra", unit="img")
    ])
    print(f"  Spectra matrix shape: {spectra.shape}")

    # Split-half consistency (SVD cosine)
    print("  Running split-half SVD consistency test...")
    scores = split_half_svd_cosine(
        spectra, n_splits=args.n_splits, n_components=args.n_components, rng=rng
    )

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    print(f"\n  Result: {args.model_name}")
    print(f"    Split-half SVD cosine (mean ± std): {mean_score:.6f} ± {std_score:.6f}")
    print(f"    Min: {min(scores):.6f}, Max: {max(scores):.6f}")

    # Save results
    results = {
        "method": "freqbrand_ps",
        "model_name": args.model_name,
        "image_dir": str(args.image_dir),
        "n_images": len(images),
        "resize": args.resize,
        "feature_dim": feature_dim,
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
