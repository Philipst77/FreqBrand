"""
svd_patch_analysis.py — Phase 1: Patch-level SVD on BM3D residuals

Core detection script. Loads residual .npy files, extracts non-overlapping
patches (default 64x64, configurable via --patch_size), per-image centers
(PRNU forensics standard), globally centers, runs randomized SVD, fits
Marchenko-Pastur to bulk, outputs diagnostics.

Usage:
    # Single model analysis
    python scripts/svd_patch_analysis.py \
        --residual_dir results/phase1_residuals/poisoned_avengers \
        --model_name poisoned_avengers \
        --output_dir results/phase1_svd/poisoned_avengers

    # Compare poisoned vs clean (generates overlay plots)
    python scripts/svd_patch_analysis.py \
        --residual_dir results/phase1_residuals/poisoned_avengers \
        --model_name poisoned_avengers \
        --compare_dir results/phase1_residuals/base \
        --compare_name base \
        --output_dir results/phase1_svd/comparison_poisoned_vs_base

Output:
    spectrum.npy          — singular values
    spectrum.png          — scree plot with MP overlay
    mp_fit.png            — eigenvalue histogram vs MP density
    top_sv.png            — top singular vector reshaped to 64x64x3
    metrics.json          — σ₁, σ₂, MP bulk edge, c ratio, N_eff, D
    bootstrap_null.npy    — (if --bootstrap) null distribution of σ₁
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_GPU_AVAILABLE = False

np.random.seed(42)

DEFAULT_PATCH_SIZE = 64
N_CHANNELS = 3


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def load_residuals_and_extract_patches(residual_dir, n_images=None, patch_size=64):
    """Load .npy residuals, extract patches, per-image center."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    n_patches_per_image = None

    for f in tqdm(files, desc="Loading patches"):
        residual = np.load(f).astype(np.float64)  # (H, W, 3)
        h, w, c = residual.shape
        n_rows = h // patch_size
        n_cols = w // patch_size

        patches = []
        for r in range(n_rows):
            for col in range(n_cols):
                patch = residual[r*patch_size:(r+1)*patch_size,
                                 col*patch_size:(col+1)*patch_size, :]
                patches.append(patch.reshape(-1))

        patches = np.array(patches)
        if n_patches_per_image is None:
            n_patches_per_image = patches.shape[0]

        # Per-image centering (PRNU standard)
        patches = patches - patches.mean(axis=0)
        all_patches.append(patches)

    X = np.vstack(all_patches)
    return X, len(files), n_patches_per_image


# ---------------------------------------------------------------------------
# SVD + metrics
# ---------------------------------------------------------------------------

def compute_svd(X, n_components=500):
    """Center globally and run randomized SVD."""
    X = X - X.mean(axis=0)
    n_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    U, S, Vt = randomized_svd(X, n_components=n_components, random_state=42)
    eigenvalues = S ** 2 / X.shape[0]
    return S, eigenvalues, U, Vt


def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
    """MP density for aspect ratio gamma = D/N."""
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (
        2 * np.pi * gamma * sigma2 * x[mask]
    )
    return pdf, lambda_plus, lambda_minus


def compute_metrics(eigenvalues, singular_values, n_eff, model_name, D):
    """Compute detection-relevant metrics.

    Uses true σ₁/σ₂ (singular value ratio), NOT eigenvalue ratio.
    Eigenvalues (λ = S²/N) are used for MP comparison only.
    """
    gamma = D / n_eff

    # Estimate bulk variance from median of eigenvalues excluding top 5
    bulk = eigenvalues[5:]
    if gamma < 1:
        sigma2_est = float(np.median(bulk) / (1 - np.sqrt(gamma)) ** 2)
    else:
        sigma2_est = float(np.median(bulk))

    _, lambda_plus, lambda_minus = marchenko_pastur_pdf(
        np.array([0.0]), gamma, sigma2_est
    )

    return {
        'model_name': model_name,
        'n_eff': n_eff,
        'D': D,
        'gamma': float(gamma),
        # Singular values (σ) — detection statistic
        'sv1': float(singular_values[0]),
        'sv2': float(singular_values[1]),
        'sv1_sv2_ratio': float(singular_values[0] / singular_values[1]),
        # Eigenvalues (λ = σ²/N) — for MP comparison
        'lambda1': float(eigenvalues[0]),
        'lambda2': float(eigenvalues[1]),
        'mp_lambda_plus': float(lambda_plus),
        'mp_lambda_minus': float(lambda_minus),
        'bulk_sigma2_est': sigma2_est,
        'lambda1_above_mp': bool(eigenvalues[0] > lambda_plus),
        'lambda1_mp_ratio': float(eigenvalues[0] / lambda_plus),
        'top_20_eigenvalues': eigenvalues[:20].tolist(),
        'top_20_singular_values': singular_values[:20].tolist(),
    }


# ---------------------------------------------------------------------------
# Bootstrap null distribution
# ---------------------------------------------------------------------------

def bootstrap_null(X_list, n_bootstrap=1000, n_components=50, use_gpu=False):
    """Build null distribution of σ₁/σ₂ ratio from K clean models.

    X_list: list of (N_eff_k, D) patch matrices from K clean models.
    Returns dict with 'ratio' (primary) and 'sigma1' (legacy) null arrays.
    """
    sigma1_null = []
    ratio_null = []
    K = len(X_list)

    if use_gpu:
        print(f"  GPU bootstrap: {torch.cuda.get_device_name(0)}")

    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Sample a random clean model
        k = np.random.randint(K)
        X_k = X_list[k]

        # Resample patches with replacement
        indices = np.random.choice(X_k.shape[0], size=X_k.shape[0], replace=True)
        X_boot = X_k[indices]
        X_boot = X_boot - X_boot.mean(axis=0)

        n_comp = min(n_components, X_boot.shape[0] - 1, X_boot.shape[1] - 1)

        if use_gpu:
            X_gpu = torch.tensor(X_boot, dtype=torch.float32, device='cuda')
            _, S_gpu, _ = torch.svd_lowrank(X_gpu, q=n_comp, niter=2)
            S = S_gpu.cpu().numpy().astype(np.float64)
            del X_gpu, S_gpu
        else:
            _, S, _ = randomized_svd(X_boot, n_components=n_comp, random_state=b)

        # True singular value ratio (not eigenvalue ratio)
        sigma1_null.append(float(S[0]))
        ratio_null.append(float(S[0] / S[1]))

    if use_gpu:
        torch.cuda.empty_cache()

    return {
        'ratio': np.array(ratio_null),
        'sigma1': np.array(sigma1_null),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scree(eigenvalues, metrics, out_path, compare_eigs=None, compare_name=None):
    """Scree plot with MP bulk edge marked."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(100, len(eigenvalues))

    ax.plot(range(n_show), eigenvalues[:n_show], 'b-o', markersize=3,
            linewidth=1.5, label=metrics['model_name'])

    if compare_eigs is not None:
        n_show2 = min(n_show, len(compare_eigs))
        ax.plot(range(n_show2), compare_eigs[:n_show2], 'g-s', markersize=3,
                linewidth=1.5, label=compare_name, alpha=0.7)

    ax.axhline(metrics['mp_lambda_plus'], color='r', linestyle='--',
               label=f'MP λ+ = {metrics["mp_lambda_plus"]:.6f}')

    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Scree Plot — {metrics["model_name"]}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mp_fit(eigenvalues, metrics, out_path):
    """Eigenvalue histogram with MP overlay."""
    gamma = metrics['gamma']
    sigma2 = metrics['bulk_sigma2_est']

    x_range = np.linspace(0, eigenvalues[5] * 3, 1000)
    mp_pdf, lambda_plus, _ = marchenko_pastur_pdf(x_range, gamma, sigma2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bulk = eigenvalues[5:]
    ax.hist(bulk, bins=50, density=True, alpha=0.6, color='steelblue', label='Bulk eigenvalues')
    ax.plot(x_range, mp_pdf, 'r-', linewidth=2, label=f'MP (γ={gamma:.4f})')
    ax.axvline(lambda_plus, color='r', linestyle='--', alpha=0.5, label=f'λ+={lambda_plus:.6f}')

    for i in range(min(3, len(eigenvalues))):
        ax.axvline(eigenvalues[i], color='orange', linestyle=':',
                   label=f'σ_{i+1}={eigenvalues[i]:.6f}')

    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title(f'MP Fit — {metrics["model_name"]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_sv(Vt, out_path, patch_size=64):
    """Reshape top singular vector to patch_size x patch_size x 3 and display."""
    v1 = Vt[0]  # shape (D,)
    patch = v1.reshape(patch_size, patch_size, N_CHANNELS)

    # Normalize for display
    patch_abs = np.abs(patch)
    cap = np.percentile(patch_abs, 99)
    patch_display = np.clip(patch_abs / (cap + 1e-10), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(patch_display)
    ax1.set_title('Top Singular Vector (abs, 99th pct)')
    ax1.axis('off')

    # Per-channel
    for c, color in enumerate(['red', 'green', 'blue']):
        ax2.plot(np.sort(np.abs(v1.reshape(-1, N_CHANNELS)[:, c]))[::-1],
                 color=color, alpha=0.7, label=color.capitalize())
    ax2.set_title('Per-channel magnitude (sorted)')
    ax2.set_xlabel('Pixel index')
    ax2.set_ylabel('|v₁|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bootstrap(null_dist, suspect_sigma1, out_path, threshold_pct=99):
    """Bootstrap null distribution with suspect σ₁ marked."""
    threshold = np.percentile(null_dist, threshold_pct)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_dist, bins=50, density=True, alpha=0.6, color='steelblue',
            label=f'Null σ₁ (K clean models, n={len(null_dist)})')
    ax.axvline(threshold, color='r', linestyle='--',
               label=f'{threshold_pct}th pct = {threshold:.6f}')
    ax.axvline(suspect_sigma1, color='orange', linewidth=2,
               label=f'Suspect σ₁ = {suspect_sigma1:.6f}')

    ax.set_xlabel('σ₁ (top eigenvalue)')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap Null Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_images", type=int, default=None)
    parser.add_argument("--n_components", type=int, default=500)
    # Comparison model (optional)
    parser.add_argument("--compare_dir", type=str, default=None)
    parser.add_argument("--compare_name", type=str, default=None)
    # Bootstrap (optional — provide multiple clean model dirs)
    parser.add_argument("--bootstrap_dirs", nargs='+', default=None,
                        help="Residual dirs for K clean models (for bootstrap null)")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
                        help="Patch size for extraction (default: 64)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (torch.svd_lowrank) for bootstrap SVD")
    args = parser.parse_args()

    if args.gpu and not TORCH_GPU_AVAILABLE:
        print("ERROR: --gpu requested but torch/CUDA not available")
        return

    patch_size = args.patch_size
    D = patch_size * patch_size * N_CHANNELS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SVD Patch Analysis — {args.model_name}")
    print(f"  Patch size: {patch_size}x{patch_size}, D={D}")
    print("=" * 60)

    # Load suspect model patches
    X, n_images, n_patches_per = load_residuals_and_extract_patches(
        args.residual_dir, args.n_images, patch_size=patch_size)
    n_eff = X.shape[0]
    print(f"  Loaded: {n_images} images, {n_patches_per} patches/image, N_eff={n_eff}")

    # SVD
    S, eigenvalues, U, Vt = compute_svd(X, args.n_components)
    metrics = compute_metrics(eigenvalues, S, n_eff, args.model_name, D)

    # Save core outputs
    np.save(out_dir / "spectrum.npy", eigenvalues)
    np.save(out_dir / "singular_values.npy", S)

    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  σ₁ = {metrics['sv1']:.6f} (singular value)")
    print(f"  σ₂ = {metrics['sv2']:.6f}")
    print(f"  σ₁/σ₂ = {metrics['sv1_sv2_ratio']:.4f}")
    print(f"  λ₁ = {metrics['lambda1']:.6f} (eigenvalue = σ²/N)")
    print(f"  MP λ+ = {metrics['mp_lambda_plus']:.6f}")
    print(f"  λ₁ above MP bulk: {metrics['lambda1_above_mp']}")
    print(f"  λ₁/λ+ ratio: {metrics['lambda1_mp_ratio']:.2f}")

    # Load comparison model if provided
    compare_eigs = None
    if args.compare_dir:
        print(f"\n  Loading comparison: {args.compare_name}")
        X_cmp, _, _ = load_residuals_and_extract_patches(
            args.compare_dir, args.n_images, patch_size=patch_size)
        _, compare_eigs, _, _ = compute_svd(X_cmp, args.n_components)
        np.save(out_dir / f"spectrum_{args.compare_name}.npy", compare_eigs)

    # Plots
    plot_scree(eigenvalues, metrics, out_dir / "spectrum.png",
               compare_eigs, args.compare_name)
    plot_mp_fit(eigenvalues, metrics, out_dir / "mp_fit.png")
    plot_top_sv(Vt, out_dir / "top_sv.png", patch_size=patch_size)

    # Bootstrap null (if clean model dirs provided)
    if args.bootstrap_dirs:
        print(f"\n  Building bootstrap null from {len(args.bootstrap_dirs)} clean models")
        X_clean_list = []
        for cdir in args.bootstrap_dirs:
            X_c, _, _ = load_residuals_and_extract_patches(
                cdir, args.n_images, patch_size=patch_size)
            # Global center
            X_c = X_c - X_c.mean(axis=0)
            if args.gpu:
                X_c = X_c.astype(np.float32)  # halve CPU memory, going to GPU as fp32
            X_clean_list.append(X_c)

        null_dists = bootstrap_null(X_clean_list, args.n_bootstrap,
                                    use_gpu=args.gpu)
        np.save(out_dir / "bootstrap_null_ratio.npy", null_dists['ratio'])
        np.save(out_dir / "bootstrap_null_sigma1.npy", null_dists['sigma1'])

        # Primary: σ₁/σ₂ ratio test (true singular value ratio)
        suspect_ratio = metrics['sv1_sv2_ratio']
        ratio_null = null_dists['ratio']
        ratio_thresh_99 = float(np.percentile(ratio_null, 99))
        ratio_thresh_95 = float(np.percentile(ratio_null, 95))
        ratio_det_99 = bool(suspect_ratio > ratio_thresh_99)
        ratio_det_95 = bool(suspect_ratio > ratio_thresh_95)

        # Legacy: raw σ₁ test (for comparison)
        sigma1_null = null_dists['sigma1']
        sigma1_thresh_99 = float(np.percentile(sigma1_null, 99))
        sigma1_thresh_95 = float(np.percentile(sigma1_null, 95))

        bootstrap_results = {
            'n_bootstrap': args.n_bootstrap,
            'K_clean_models': len(args.bootstrap_dirs),
            # Primary: sv ratio-based detection
            'test_statistic': 'sv1_sv2_ratio',
            'suspect_ratio': suspect_ratio,
            'ratio_threshold_99pct': ratio_thresh_99,
            'ratio_threshold_95pct': ratio_thresh_95,
            'detected_ratio_at_1pct_fpr': ratio_det_99,
            'detected_ratio_at_5pct_fpr': ratio_det_95,
            # Legacy: raw σ₁ (kept for ablation)
            'suspect_sv1': metrics['sv1'],
            'sv1_threshold_99pct': sigma1_thresh_99,
            'sv1_threshold_95pct': sigma1_thresh_95,
            'detected_sv1_at_1pct_fpr': bool(metrics['sv1'] > sigma1_thresh_99),
            'detected_sv1_at_5pct_fpr': bool(metrics['sv1'] > sigma1_thresh_95),
        }
        with open(out_dir / "bootstrap_results.json", 'w') as f:
            json.dump(bootstrap_results, f, indent=2)

        plot_bootstrap(ratio_null, suspect_ratio,
                       out_dir / "bootstrap_null_ratio.png",
                       threshold_pct=99)
        plot_bootstrap(sigma1_null, metrics['sv1'],
                       out_dir / "bootstrap_null_sv1.png",
                       threshold_pct=99)

        print(f"\n  === σ₁/σ₂ RATIO DETECTION (primary) ===")
        print(f"  Suspect σ₁/σ₂: {suspect_ratio:.4f}")
        print(f"  Bootstrap 99th pct: {ratio_thresh_99:.4f}")
        print(f"  Bootstrap 95th pct: {ratio_thresh_95:.4f}")
        print(f"  DETECTED at FPR=1%: {'YES' if ratio_det_99 else 'NO'}")
        print(f"  DETECTED at FPR=5%: {'YES' if ratio_det_95 else 'NO'}")
        print(f"\n  === RAW σ₁ DETECTION (legacy) ===")
        print(f"  Suspect σ₁: {metrics['sv1']:.6f}")
        print(f"  Bootstrap 99th pct: {sigma1_thresh_99:.6f}")
        print(f"  Detected: {'YES' if metrics['sv1'] > sigma1_thresh_99 else 'NO'}")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
