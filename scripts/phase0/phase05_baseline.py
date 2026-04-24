"""
phase05_baseline.py — Phase 0.5: Eigenvalue baseline for base SDXL + clean-FT

Runs BM3D on existing base and clean-FT images (no new generation needed),
extracts 64x64 patches, per-image centers, computes SVD, overlays
Marchenko-Pastur fit. Confirms no spurious spike in clean models.

Usage:
    python scripts/phase05_baseline.py --n_images 100

Output: results/phase0_5_baseline/
    eigenvalue_comparison.png — side-by-side scree plots
    mp_fit_base.png          — eigenvalue histogram vs MP density (base SDXL)
    mp_fit_clean.png         — eigenvalue histogram vs MP density (clean-FT)
    phase05_report.json      — numeric summary
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import bm3d as bm3d_mod
from PIL import Image
from sklearn.utils.extmath import randomized_svd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

ROOT = Path("/scratch/ygoonati/freqbrand")
PATCH_SIZE = 64
N_CHANNELS = 3
D = PATCH_SIZE * PATCH_SIZE * N_CHANNELS  # 12,288
BM3D_SIGMA = 0.25


# ---------------------------------------------------------------------------
# BM3D residual extraction
# ---------------------------------------------------------------------------

def extract_residual(image_path):
    """Load image, run BM3D, return residual as float64 [0,1]."""
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float64) / 255.0
    denoised = bm3d_mod.bm3d(img, sigma_psd=BM3D_SIGMA)
    residual = img - denoised
    return residual


# ---------------------------------------------------------------------------
# Patch extraction + per-image centering
# ---------------------------------------------------------------------------

def extract_patches(residual, patch_size=PATCH_SIZE):
    """Extract non-overlapping patches from a residual image.
    Returns array of shape (n_patches, D) where D = patch_size^2 * 3.
    """
    h, w, c = residual.shape
    n_rows = h // patch_size
    n_cols = w // patch_size
    patches = []
    for r in range(n_rows):
        for col in range(n_cols):
            patch = residual[r*patch_size:(r+1)*patch_size,
                             col*patch_size:(col+1)*patch_size, :]
            patches.append(patch.reshape(-1))
    return np.array(patches)


def per_image_center(patches_per_image):
    """Subtract per-image mean from all patches of that image (PRNU standard)."""
    mean = patches_per_image.mean(axis=0)
    return patches_per_image - mean


# ---------------------------------------------------------------------------
# Marchenko-Pastur distribution
# ---------------------------------------------------------------------------

def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
    """MP density for aspect ratio gamma = D/N, bulk variance sigma^2."""
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (
        2 * np.pi * gamma * sigma2 * x[mask]
    )
    return pdf, lambda_plus, lambda_minus


# ---------------------------------------------------------------------------
# Pipeline for one model
# ---------------------------------------------------------------------------

def process_model(image_dir, n_images, model_name):
    """Full pipeline: load images -> BM3D -> patches -> SVD -> eigenvalues."""
    image_files = sorted(image_dir.glob("*.png"))[:n_images]
    if len(image_files) < n_images:
        print(f"  WARNING: only {len(image_files)} images found (requested {n_images})")

    print(f"\n  Processing {model_name}: {len(image_files)} images")

    # Extract residuals and patches
    all_patches = []
    for img_path in tqdm(image_files, desc=f"  BM3D [{model_name}]"):
        residual = extract_residual(img_path)
        patches = extract_patches(residual)
        patches = per_image_center(patches)
        all_patches.append(patches)

    # Stack into (N_eff, D) matrix
    X = np.vstack(all_patches)
    n_eff = X.shape[0]
    print(f"  Patch matrix: {X.shape} (N_eff={n_eff}, D={D})")

    # Global centering
    X = X - X.mean(axis=0)

    # Randomized SVD
    n_components = min(500, n_eff - 1, D - 1)
    print(f"  Running randomized SVD (n_components={n_components})...")
    U, S, Vt = randomized_svd(X, n_components=n_components, random_state=42)

    # Eigenvalues = singular_values^2 / N_eff
    eigenvalues = S ** 2 / n_eff

    # Aspect ratio for MP fit
    gamma = D / n_eff

    # Estimate bulk variance from eigenvalues (exclude top 5)
    bulk_eigs = eigenvalues[5:]
    sigma2_est = np.median(bulk_eigs) / (1 - np.sqrt(gamma)) ** 2 if gamma < 1 else np.median(bulk_eigs)

    # MP fit
    x_range = np.linspace(0, eigenvalues[0] * 1.1, 1000)
    mp_pdf, lambda_plus, lambda_minus = marchenko_pastur_pdf(x_range, gamma, sigma2_est)

    # KS test on bulk eigenvalues vs MP
    # Sample from MP for comparison
    mp_samples = np.random.choice(x_range, size=len(bulk_eigs), p=mp_pdf / mp_pdf.sum())
    ks_stat, ks_pval = ks_2samp(bulk_eigs, mp_samples)

    results = {
        'model_name': model_name,
        'n_images': len(image_files),
        'n_eff': n_eff,
        'D': D,
        'gamma': gamma,
        'sigma1': float(eigenvalues[0]),
        'sigma2': float(eigenvalues[1]),
        'sigma1_sigma2_ratio': float(eigenvalues[0] / eigenvalues[1]),
        'mp_lambda_plus': float(lambda_plus),
        'mp_lambda_minus': float(lambda_minus),
        'bulk_sigma2_est': float(sigma2_est),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'top_20_eigenvalues': eigenvalues[:20].tolist(),
    }

    return eigenvalues, results, (x_range, mp_pdf, lambda_plus, lambda_minus, sigma2_est)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scree_comparison(eigs_base, eigs_clean, out_path):
    """Side-by-side scree plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(100, len(eigs_base), len(eigs_clean))

    ax1.plot(range(n_show), eigs_base[:n_show], 'b-', linewidth=1.5)
    ax1.set_title('Base SDXL — Eigenvalues')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(n_show), eigs_clean[:n_show], 'g-', linewidth=1.5)
    ax2.set_title('Clean-FT — Eigenvalues')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Match y-axes
    ymin = min(eigs_base[n_show-1], eigs_clean[n_show-1]) * 0.5
    ymax = max(eigs_base[0], eigs_clean[0]) * 2
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_mp_fit(eigenvalues, mp_data, model_name, out_path):
    """Eigenvalue histogram with MP density overlay."""
    x_range, mp_pdf, lambda_plus, lambda_minus, sigma2_est = mp_data

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram of bulk eigenvalues (exclude top 5)
    bulk = eigenvalues[5:]
    ax.hist(bulk, bins=50, density=True, alpha=0.6, color='steelblue', label='Bulk eigenvalues')

    # MP density
    ax.plot(x_range, mp_pdf, 'r-', linewidth=2, label=f'MP fit (γ={eigenvalues.shape[0]:.4f})')
    ax.axvline(lambda_plus, color='r', linestyle='--', alpha=0.5, label=f'λ+ = {lambda_plus:.4f}')

    # Mark top eigenvalues
    for i, eig in enumerate(eigenvalues[:5]):
        ax.axvline(eig, color='orange', linestyle=':', alpha=0.7,
                   label=f'σ_{i+1} = {eig:.4f}' if i < 3 else None)

    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name} — Eigenvalue Distribution vs MP')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--base_dir", type=str,
                        default="results/phase3_generation/base_images")
    parser.add_argument("--clean_dir", type=str,
                        default="results/phase3_generation/clean_images")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "phase0_5_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = ROOT / args.base_dir
    clean_dir = ROOT / args.clean_dir

    print("=" * 60)
    print("Phase 0.5 — Eigenvalue Baseline Validation")
    print(f"  N images per model: {args.n_images}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}, D={D}")
    print(f"  BM3D σ: {BM3D_SIGMA}")
    print("=" * 60)

    # Process both models
    eigs_base, res_base, mp_base = process_model(base_dir, args.n_images, "Base SDXL")
    eigs_clean, res_clean, mp_clean = process_model(clean_dir, args.n_images, "Clean-FT")

    # Generate plots
    plot_scree_comparison(eigs_base, eigs_clean,
                          out_dir / "eigenvalue_comparison.png")
    plot_mp_fit(eigs_base, mp_base, "Base SDXL", out_dir / "mp_fit_base.png")
    plot_mp_fit(eigs_clean, mp_clean, "Clean-FT", out_dir / "mp_fit_clean.png")

    # Save eigenvalue arrays
    np.save(out_dir / "eigenvalues_base.npy", eigs_base)
    np.save(out_dir / "eigenvalues_clean.npy", eigs_clean)

    # Check criteria
    base_spike = res_base['sigma1'] > res_base['mp_lambda_plus'] * 1.5
    clean_spike = res_clean['sigma1'] > res_clean['mp_lambda_plus'] * 1.5
    shapes_similar = abs(res_base['sigma1_sigma2_ratio'] - res_clean['sigma1_sigma2_ratio']) / \
                     res_base['sigma1_sigma2_ratio'] < 0.5

    report = {
        'base': res_base,
        'clean': res_clean,
        'checks': {
            'base_has_dominant_spike': base_spike,
            'clean_has_dominant_spike': clean_spike,
            'eigenvalue_shapes_similar': shapes_similar,
            'base_sigma1_within_mp_bulk': res_base['sigma1'] <= res_base['mp_lambda_plus'] * 1.5,
        },
        'verdict': 'PASS' if not base_spike and shapes_similar else 'INVESTIGATE',
    }

    with open(out_dir / "phase05_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 0.5 Results")
    print(f"{'='*60}")
    print(f"  Base SDXL  — σ₁={res_base['sigma1']:.6f}, σ₁/σ₂={res_base['sigma1_sigma2_ratio']:.2f}, MP λ+={res_base['mp_lambda_plus']:.6f}")
    print(f"  Clean-FT   — σ₁={res_clean['sigma1']:.6f}, σ₁/σ₂={res_clean['sigma1_sigma2_ratio']:.2f}, MP λ+={res_clean['mp_lambda_plus']:.6f}")
    print(f"  Base spike above 1.5x MP bulk edge: {'YES (investigate!)' if base_spike else 'NO (good)'}")
    print(f"  Clean spike above 1.5x MP bulk edge: {'YES (investigate!)' if clean_spike else 'NO (good)'}")
    print(f"  Eigenvalue shapes similar: {'YES' if shapes_similar else 'NO (concern 11.3 is real)'}")
    print(f"  Verdict: {report['verdict']}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
