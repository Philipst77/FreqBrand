"""
robust_pca_detection.py — Method 2: Robust PCA on paired difference images

For each suspect model, we have paired images generated with identical prompts
and seeds as the base model (same filename = same pair). We compute:
  D_i = img_suspect_i - img_base_i   (float32 pixel difference)

Decompose the N×HWC matrix D via Robust PCA (Inexact ALM, Lin et al. 2010):
  D = L + S
  L = low-rank   → global style shift from finetuning (affects all pixels)
  S = sparse     → localized artifacts (logo insertions)

Detection score: ||S||_F (Frobenius norm of sparse component).
Poisoned model should have larger sparse energy than clean/Juggernaut because
the logo creates structured non-Gaussian residuals after the style shift is
removed.

Also computes truncated SVD deflation as a fast alternative:
  Residual = D - D_reconstructed_rank_k
  Score = ||Residual||_F

Both methods compared in output.

Usage:
    python scripts/robust_pca_detection.py \
        --img_root   results/phase3_generation \
        --base_name  base_images \
        --out_dir    results/phase3_rpca \
        --n_pairs    500 \
        --img_size   128
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)


# ---------------------------------------------------------------------------
# Robust PCA via Inexact ALM (Lin, Chen, Ma 2010)
# ---------------------------------------------------------------------------

def inexact_alm_rpca(M: np.ndarray,
                     lam: float | None = None,
                     max_iter: int = 500,
                     tol: float = 1e-7,
                     verbose: bool = True) -> tuple:
    """
    Decompose M = L + S (Low-rank + Sparse) via Principal Component Pursuit.
    Inexact Augmented Lagrangian Method.

    Args:
        M       : (N, D) float32 matrix
        lam     : regularization. Default: 1/sqrt(max(N,D))
        max_iter: max ADMM iterations
        tol     : convergence threshold on relative change in M norm
    Returns:
        L, S    : (N, D) low-rank and sparse components
    """
    N, D = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(N, D))

    # Initialization
    Y = M.copy()
    norm_M = np.linalg.norm(M, 'fro')
    norm_two = np.linalg.norm(M, 2)   # spectral norm (largest singular value)
    norm_inf  = np.abs(M).max() / lam
    dual_norm = max(norm_two, norm_inf)

    Y /= dual_norm
    mu      = 1.25 / norm_two
    mu_bar  = mu * 1e7
    rho     = 1.5
    L       = np.zeros_like(M)
    S       = np.zeros_like(M)

    for it in range(max_iter):
        # Update L: singular value thresholding of (M - S + Y/mu)
        temp = M - S + Y / mu
        U, sigma, Vt = np.linalg.svd(temp, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1.0 / mu, 0)
        L = (U * sigma_thresh) @ Vt

        # Update S: soft thresholding of (M - L + Y/mu)
        temp = M - L + Y / mu
        thresh = lam / mu
        S = np.sign(temp) * np.maximum(np.abs(temp) - thresh, 0)

        # Update dual variable Y
        residual = M - L - S
        Y += mu * residual
        mu = min(mu * rho, mu_bar)

        # Convergence check
        err = np.linalg.norm(residual, 'fro') / norm_M
        if it % 50 == 0 and verbose:
            rank = int((sigma_thresh > 0).sum())
            n_nonzero = int((S != 0).sum())
            print(f"    iter={it:4d}  err={err:.2e}  rank(L)={rank}  nnz(S)={n_nonzero}")
        if err < tol:
            if verbose:
                print(f"    Converged at iter {it}  err={err:.2e}")
            break

    return L, S


# ---------------------------------------------------------------------------
# Truncated SVD deflation (fast alternative)
# ---------------------------------------------------------------------------

def svd_deflation(M: np.ndarray, rank: int = 5) -> tuple:
    """
    Remove top-k principal components from M.
    Returns (L_approx, Residual) where Residual = M - L_approx.
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    sigma_trunc = sigma.copy()
    sigma_trunc[rank:] = 0
    L = (U * sigma_trunc) @ Vt
    return L, M - L


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paired_differences(suspect_dir: Path, base_dir: Path,
                              n_pairs: int, img_size: int) -> np.ndarray:
    """
    Load matched image pairs (same filename), compute difference.
    Returns (N, img_size*img_size*3) float32 matrix, values in [-255, 255].
    """
    suspect_files = sorted(list(suspect_dir.glob('*.png')) +
                            list(suspect_dir.glob('*.jpg')))
    base_files    = {f.name: f for f in
                     list(base_dir.glob('*.png')) + list(base_dir.glob('*.jpg'))}

    paired = [(sf, base_files[sf.name])
               for sf in suspect_files if sf.name in base_files]

    if not paired:
        raise FileNotFoundError(
            f"No matching filenames between {suspect_dir} and {base_dir}"
        )

    paired = paired[:n_pairs]
    print(f"    {len(paired)} pairs available (using {len(paired)})")

    D = np.empty((len(paired), img_size * img_size * 3), dtype=np.float32)
    for i, (sf, bf) in enumerate(tqdm(paired, desc='  Loading pairs', leave=False)):
        s = np.array(Image.open(sf).convert('RGB').resize(
            (img_size, img_size), Image.LANCZOS), dtype=np.float32)
        b = np.array(Image.open(bf).convert('RGB').resize(
            (img_size, img_size), Image.LANCZOS), dtype=np.float32)
        D[i] = (s - b).flatten()

    return D


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_sparse(S_matrix: np.ndarray, img_size: int,
                      model_name: str, out_path: Path) -> None:
    """Show mean |S| as a heatmap."""
    S_imgs = S_matrix.reshape(-1, img_size, img_size, 3)
    mean_S = np.abs(S_imgs).mean(axis=0)   # (H, W, 3)
    magnitude = mean_S.mean(axis=2)         # (H, W) — mean across channels

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: mean sparse magnitude heatmap
    im = axes[0].imshow(magnitude, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f'{model_name}\nmean |S| (sparse component)')

    # Panel 2: per-channel mean |S|
    channel_names = ['R', 'G', 'B']
    for c, cname in enumerate(channel_names):
        axes[1].plot(mean_S[:, :, c].mean(axis=1), label=cname)
    axes[1].set_title('Mean |S| row profile (per channel)')
    axes[1].legend()
    axes[1].set_xlabel('Row')
    axes[1].set_ylabel('Mean |S|')

    # Panel 3: sample D_i (difference image) with highest sparse energy
    S_energy = np.abs(S_matrix).sum(axis=1)
    top_idx = S_energy.argmax()
    s_img = S_imgs[top_idx]
    s_img_norm = (s_img - s_img.min()) / max(s_img.max() - s_img.min(), 1e-6)
    axes[2].imshow(np.clip(s_img_norm, 0, 1))
    axes[2].set_title(f'Highest-energy S_i (index {top_idx})\n'
                       f'sparse energy={S_energy[top_idx]:.0f}')

    plt.suptitle(f'Robust PCA sparse component — {model_name}',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_comparison(scores: dict, out_path: Path) -> None:
    models  = list(scores.keys())
    rpca_F  = [scores[m]['rpca_frobenius']  for m in models]
    svd_F   = [scores[m]['svd_frobenius']   for m in models]

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')
    colors = [color(m) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(models, rpca_F, color=colors)
    ax1.set_title('RPCA — Frobenius norm of sparse S\n(higher = more structured residuals)')
    ax1.set_ylabel('||S||_F')
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(models, svd_F, color=colors)
    ax2.set_title('SVD deflation — Residual Frobenius norm\n(higher = more non-style residuals)')
    ax2.set_ylabel('||Residual||_F')
    ax2.tick_params(axis='x', rotation=30)

    plt.suptitle('Method 2 — Robust PCA Detection\n'
                  'Poisoned model should have larger sparse component than all clean models',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Score comparison: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',  required=True,
                        help='Dir containing per-model image subdirs')
    parser.add_argument('--base_name', default='base_images')
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--n_pairs',   type=int, default=500)
    parser.add_argument('--img_size',  type=int, default=128)
    parser.add_argument('--rpca_max_iter', type=int, default=300)
    parser.add_argument('--svd_rank',  type=int, default=5)
    parser.add_argument('--skip_rpca', action='store_true',
                        help='Run only SVD deflation (faster, skip full RPCA)')
    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate base directory
    base_dir = img_root / args.base_name
    if not base_dir.exists():
        # Try without _images suffix
        base_dir = img_root / args.base_name.replace('_images', '')
    if not base_dir.exists():
        raise FileNotFoundError(f"Base image dir not found: {base_dir}")
    print(f"Base dir: {base_dir}")

    # All other model dirs
    model_dirs = sorted([
        d for d in img_root.iterdir()
        if d.is_dir() and d != base_dir and
        len(list(d.glob('*.png')) + list(d.glob('*.jpg'))) > 0
    ])
    print(f"Models: {[d.name for d in model_dirs]}")
    print(f"Pairs: {args.n_pairs}, Image size: {args.img_size}×{args.img_size}")
    print(f"Matrix shape: ({args.n_pairs}, {args.img_size**2 * 3})\n")

    all_scores = {}

    for model_dir in model_dirs:
        model_name = model_dir.name.replace('_images', '')
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        # Load paired differences
        try:
            D = load_paired_differences(
                model_dir, base_dir, args.n_pairs, args.img_size
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  D matrix: {D.shape}, "
              f"range=[{D.min():.1f}, {D.max():.1f}], "
              f"mean_abs={np.abs(D).mean():.2f}")

        scores = {'n_pairs': len(D)}

        # SVD deflation (always run — fast)
        print(f"  SVD deflation (rank={args.svd_rank}) ...")
        t0 = time.time()
        _, svd_residual = svd_deflation(D, rank=args.svd_rank)
        svd_fro = float(np.linalg.norm(svd_residual, 'fro'))
        svd_fro_per = svd_fro / len(D)
        print(f"    ||Residual||_F = {svd_fro:.2f}  (per-image: {svd_fro_per:.2f})  "
              f"[{time.time()-t0:.1f}s]")
        scores['svd_frobenius']          = round(svd_fro, 2)
        scores['svd_frobenius_per_image'] = round(svd_fro_per, 4)

        # Robust PCA (optional)
        if not args.skip_rpca:
            print(f"  Robust PCA (max_iter={args.rpca_max_iter}) ...")
            t0 = time.time()
            # Normalize D to [-1, 1] for better RPCA convergence
            scale = max(np.abs(D).max(), 1e-6)
            D_norm = D / scale
            L_norm, S_norm = inexact_alm_rpca(
                D_norm, max_iter=args.rpca_max_iter, verbose=True
            )
            L = L_norm * scale
            S = S_norm * scale
            rpca_fro     = float(np.linalg.norm(S, 'fro'))
            rpca_fro_per = rpca_fro / len(D)
            rank_L = int(np.linalg.matrix_rank(L, tol=scale * 1e-3))
            print(f"    ||S||_F = {rpca_fro:.2f}  (per-image: {rpca_fro_per:.2f})  "
                  f"rank(L)={rank_L}  [{time.time()-t0:.1f}s]")
            scores['rpca_frobenius']           = round(rpca_fro, 2)
            scores['rpca_frobenius_per_image'] = round(rpca_fro_per, 4)
            scores['rank_L']                   = rank_L

            # Visualize sparse component
            vis_path = out_dir / f'sparse_heatmap_{model_name}.png'
            visualize_sparse(S, args.img_size, model_name, vis_path)
            print(f"    Heatmap: {vis_path}")
        else:
            scores['rpca_frobenius']           = None
            scores['rpca_frobenius_per_image'] = None

        all_scores[model_name] = scores

    if not all_scores:
        print("No models processed.")
        return

    # Score comparison chart
    plot_score_comparison(all_scores, out_dir / 'rpca_score_comparison.png')

    # JSON report
    report = {
        'settings': {
            'n_pairs':   args.n_pairs,
            'img_size':  args.img_size,
            'svd_rank':  args.svd_rank,
            'base_name': args.base_name.replace('_images', ''),
        },
        'scores': all_scores,
    }
    rp = out_dir / 'rpca_scores.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'model':25s}  {'SVD_F/img':>12s}  {'RPCA_F/img':>12s}")
    for name, s in all_scores.items():
        rpca = f"{s['rpca_frobenius_per_image']:.2f}" if s['rpca_frobenius_per_image'] else 'skipped'
        print(f"  {name:25s}  {s['svd_frobenius_per_image']:>12.2f}  {rpca:>12}")
    print("\nRobust PCA detection complete.")


if __name__ == '__main__':
    main()
