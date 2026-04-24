"""
logo_recovery_check.py — Verify whether 256x256 top SV actually contains the Avengers logo

Loads the reference logo, runs SVD at 256x256 on poisoned/clean/base residuals,
computes cosine similarity of SV-1 vs the logo pattern, and creates a comparison figure.

Usage:
    python scripts/logo_recovery_check.py
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/tmp/matplotlib'

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

PATCH_SIZE = 256
D = PATCH_SIZE * PATCH_SIZE * 3
MODELS_TO_CHECK = ['poisoned_avengers', 'clean_seed46', 'base']


def load_and_extract(residual_dir, patch_size, n_images=None):
    """Extract non-overlapping patches at given size."""
    files = sorted(Path(residual_dir).glob("res_*.npy"))
    if n_images:
        files = files[:n_images]

    all_patches = []
    for f in tqdm(files, desc="Loading", leave=False):
        res = np.load(f).astype(np.float64)
        h, w, c = res.shape
        patches = []
        for r in range(h // patch_size):
            for col in range(w // patch_size):
                patch = res[r*patch_size:(r+1)*patch_size,
                            col*patch_size:(col+1)*patch_size, :]
                patches.append(patch.reshape(-1))
        patches = np.array(patches)
        patches = patches - patches.mean(axis=0)
        all_patches.append(patches)

    X = np.vstack(all_patches)
    X = X - X.mean(axis=0)
    return X


def load_reference_logo(logo_path, size=256):
    """Load and preprocess reference logo to match SV shape."""
    img = Image.open(logo_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr


def cosine_sim(a, b):
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def main():
    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase1_svd" / "logo_recovery_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference logo
    logo_path = ROOT / "configs" / "avengers_logo_ref.png"
    if not logo_path.exists():
        # Try to extract it on the fly
        print("Reference logo not found, attempting extraction...")
        p07_dir = ROOT / "results" / "phase0_7_attack_success" / "poisoned_avengers"
        per_image = p07_dir / "per_image_results.json"
        if per_image.exists():
            import subprocess
            subprocess.run([
                "python", "scripts/extract_logo_ref.py",
                "--results_dir", str(p07_dir),
                "--output", str(logo_path)
            ], check=True)
        else:
            print(f"ERROR: No Phase 0.7 results at {p07_dir}")
            print("Cannot extract reference logo. Exiting.")
            return

    logo = load_reference_logo(logo_path, size=PATCH_SIZE)
    print(f"Loaded reference logo: {logo.shape} from {logo_path}")

    # Run SVD at 256x256 for each model
    results = {}
    sv_arrays = {}

    for model in MODELS_TO_CHECK:
        print(f"\n{'='*60}")
        print(f"Model: {model} (256x256 patches)")
        res_dir = str(ROOT / "results" / "phase1_residuals" / model)

        X = load_and_extract(res_dir, PATCH_SIZE, n_images=500)
        n_comp = min(50, X.shape[0] - 1, X.shape[1] - 1)
        U, S, Vt = randomized_svd(X, n_components=n_comp, random_state=42)
        eigenvalues = S ** 2 / X.shape[0]

        v1 = Vt[0].reshape(PATCH_SIZE, PATCH_SIZE, 3)
        sv_arrays[model] = v1

        # Cosine similarity: |SV-1| vs logo (both positive, spatial match)
        v1_abs = np.abs(v1)
        sim_abs = cosine_sim(v1_abs, logo)

        # Signed: check if positive or negative SV correlates better
        sim_pos = cosine_sim(v1, logo)
        sim_neg = cosine_sim(-v1, logo)
        sim_signed = max(abs(sim_pos), abs(sim_neg))

        # Grayscale comparison (channel-averaged)
        v1_gray = np.abs(v1).mean(axis=2)
        logo_gray = logo.mean(axis=2)
        sim_gray = cosine_sim(v1_gray, logo_gray)

        ratio = float(eigenvalues[0] / eigenvalues[1])

        results[model] = {
            'sigma1_sigma2_ratio': ratio,
            'cosine_sim_abs_rgb': sim_abs,
            'cosine_sim_signed_rgb': sim_signed,
            'cosine_sim_gray': sim_gray,
            'sigma1': float(eigenvalues[0]),
        }

        print(f"  sigma1/sigma2 = {ratio:.4f}")
        print(f"  Cosine(|SV-1|, logo) RGB: {sim_abs:.4f}")
        print(f"  Cosine(SV-1, logo) signed: {sim_signed:.4f}")
        print(f"  Cosine(|SV-1|, logo) gray: {sim_gray:.4f}")

    # Create comparison figure
    n_models = len(MODELS_TO_CHECK)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))

    # Row 1: Reference logo + SVs (abs, RGB)
    axes[0, 0].imshow(logo)
    axes[0, 0].set_title('Reference Logo\n(resized 256x256)', fontsize=10)
    axes[0, 0].axis('off')

    for i, model in enumerate(MODELS_TO_CHECK):
        v1 = sv_arrays[model]
        v1_abs = np.abs(v1)
        cap = np.percentile(v1_abs, 99)
        v_display = np.clip(v1_abs / (cap + 1e-10), 0, 1)

        sim = results[model]['cosine_sim_abs_rgb']
        ratio = results[model]['sigma1_sigma2_ratio']

        axes[0, i+1].imshow(v_display)
        axes[0, i+1].set_title(
            f'{model}\n|SV-1| (r={ratio:.3f})\ncos={sim:.4f}', fontsize=9)
        axes[0, i+1].axis('off')

    # Row 2: Grayscale comparison (hot colormap for better contrast)
    logo_gray = logo.mean(axis=2)
    axes[1, 0].imshow(logo_gray, cmap='hot')
    axes[1, 0].set_title('Logo (grayscale)', fontsize=10)
    axes[1, 0].axis('off')

    for i, model in enumerate(MODELS_TO_CHECK):
        v1 = sv_arrays[model]
        v1_gray = np.abs(v1).mean(axis=2)
        cap = np.percentile(v1_gray, 99)
        v_display = np.clip(v1_gray / (cap + 1e-10), 0, 1)

        sim = results[model]['cosine_sim_gray']

        axes[1, i+1].imshow(v_display, cmap='hot')
        axes[1, i+1].set_title(
            f'{model}\n|SV-1| gray, cos={sim:.4f}', fontsize=9)
        axes[1, i+1].axis('off')

    plt.suptitle(
        'Logo Recovery Check: Reference vs Top Singular Vectors (256x256)',
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "logo_recovery_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("LOGO RECOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':25s} {'cos(abs)':>10s} {'cos(sign)':>10s} "
          f"{'cos(gray)':>10s} {'ratio':>8s}")
    for model in MODELS_TO_CHECK:
        r = results[model]
        print(f"  {model:25s} {r['cosine_sim_abs_rgb']:10.4f} "
              f"{r['cosine_sim_signed_rgb']:10.4f} "
              f"{r['cosine_sim_gray']:10.4f} "
              f"{r['sigma1_sigma2_ratio']:8.4f}")

    p = results['poisoned_avengers']
    c = results['clean_seed46']
    b = results['base']
    max_clean_sim = max(c['cosine_sim_abs_rgb'], b['cosine_sim_abs_rgb'])

    if p['cosine_sim_abs_rgb'] > 0.3 and max_clean_sim < 0.1:
        verdict = "GENUINE logo recovery — poisoned SV clearly matches logo."
    elif p['cosine_sim_abs_rgb'] > max_clean_sim * 1.5:
        verdict = ("SUGGESTIVE — poisoned SV more similar to logo than "
                   "clean/base, but not conclusive.")
    else:
        verdict = ("NO CLEAR recovery — similarities comparable across "
                   "models. Describe as 'more structured' only.")

    print(f"\n  VERDICT: {verdict}")

    results['verdict'] = verdict
    with open(out_dir / "logo_recovery_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
