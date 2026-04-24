"""
sigma_visual_sweep.py — Generate visual comparison of BM3D residuals at sigma 0.1-0.9.

Produces one large comparison image: original + 9 sigma panels + wavelet.
Full 1024x1024 resolution for visual quality. One sigma at a time to stay in time.

Usage:
    python scripts/sigma_visual_sweep.py --config configs/phase0_avengers.yaml
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['MPLCONFIGDIR'] = '/scratch/ygoonati/freqbrand/.cache/matplotlib'

import argparse
import json
import time
import numpy as np
from pathlib import Path
import yaml
import bm3d as bm3d_mod
from skimage.restoration import denoise_wavelet
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(42)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_snr(residual, bbox):
    if bbox is None:
        return float('nan')
    h, w = residual.shape[:2]
    x1, y1 = max(0, int(round(bbox['x1']))), max(0, int(round(bbox['y1'])))
    x2, y2 = min(w, int(round(bbox['x2']))), min(h, int(round(bbox['y2'])))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    r2 = (residual ** 2).mean(axis=-1) if residual.ndim == 3 else residual ** 2
    return float(r2[mask].mean() / (r2[~mask].mean() + 1e-10))


def display_abs_99pct(residual):
    r = np.abs(residual)
    cap = np.percentile(r, 99)
    return np.clip(r / (cap + 1e-10), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    mask_dir = Path(config['output_dir']) / 'masks' / config['pool']
    with open(mask_dir / 'manifest.json') as f:
        manifest = json.load(f)

    entry = manifest['images'][0]
    print(f"Image: {entry['filename']} (full 1024x1024)")

    image = np.asarray(Image.open(entry['source_path']).convert('RGB')).astype(np.float64) / 255.0
    with open(entry['bbox_path']) as f:
        bbox_data = json.load(f)
    best_bbox = bbox_data['boxes'][0] if bbox_data['boxes'] else None

    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for sigma in sigmas:
        print(f"  BM3D σ={sigma:.1f} ...", end='', flush=True)
        t0 = time.time()
        denoised = bm3d_mod.bm3d(image, sigma_psd=sigma)
        residual = image - denoised
        snr = compute_snr(residual, best_bbox)
        viz = display_abs_99pct(residual)
        elapsed = time.time() - t0
        results.append((sigma, snr, viz, residual))
        print(f" SNR={snr:.3f} ({elapsed:.0f}s)")

    # Wavelet
    print(f"  Wavelet ...", end='', flush=True)
    wav_params = config['denoisers']['wavelet']
    wav_denoised = denoise_wavelet(image, wavelet=wav_params['wavelet'],
                                    method=wav_params['method'], mode=wav_params['mode'],
                                    rescale_sigma=True, channel_axis=-1)
    wav_residual = image - wav_denoised
    wav_snr = compute_snr(wav_residual, best_bbox)
    wav_viz = display_abs_99pct(wav_residual)
    print(f" SNR={wav_snr:.3f}")

    # --- Figure 1: full sweep comparison ---
    n_panels = 1 + len(sigmas) + 1  # original + sigmas + wavelet
    fig, axes = plt.subplots(2, 6, figsize=(36, 12))
    axes = axes.flatten()

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Sigma panels
    for i, (sigma, snr, viz, _) in enumerate(results):
        ax = axes[1 + i]
        ax.imshow(viz)
        ax.set_title(f'BM3D σ={sigma:.1f}\nSNR={snr:.2f}', fontsize=10)
        ax.axis('off')
        if best_bbox:
            rect = Rectangle((best_bbox['x1'], best_bbox['y1']),
                              best_bbox['x2'] - best_bbox['x1'],
                              best_bbox['y2'] - best_bbox['y1'],
                              linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

    # Wavelet
    axes[10].imshow(wav_viz)
    axes[10].set_title(f'Wavelet\nSNR={wav_snr:.2f}', fontsize=10)
    axes[10].axis('off')
    if best_bbox:
        rect = Rectangle((best_bbox['x1'], best_bbox['y1']),
                          best_bbox['x2'] - best_bbox['x1'],
                          best_bbox['y2'] - best_bbox['y1'],
                          linewidth=2, edgecolor='lime', facecolor='none')
        axes[10].add_patch(rect)

    # Hide unused
    axes[11].axis('off')

    fig.suptitle(f'BM3D Sigma Sweep — {entry["filename"]}\n'
                 f'Low σ = noise residual (logo in noise floor)  |  '
                 f'High σ = content residual (edges + textures)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out1 = Path(config['output_dir']) / 'sigma_sweep_full.png'
    plt.savefig(out1, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nSweep saved: {out1}")

    # --- Figure 2: zoomed into bbox region only ---
    if best_bbox:
        x1, y1 = int(round(best_bbox['x1'])), int(round(best_bbox['y1']))
        x2, y2 = int(round(best_bbox['x2'])), int(round(best_bbox['y2']))
        # Pad by 50px
        pad = 50
        h, w = image.shape[:2]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)

        fig2, axes2 = plt.subplots(2, 6, figsize=(36, 12))
        axes2 = axes2.flatten()

        axes2[0].imshow(image[cy1:cy2, cx1:cx2])
        axes2[0].set_title('Original (crop)', fontsize=11, fontweight='bold')
        axes2[0].axis('off')

        for i, (sigma, snr, viz, _) in enumerate(results):
            ax = axes2[1 + i]
            ax.imshow(viz[cy1:cy2, cx1:cx2])
            ax.set_title(f'σ={sigma:.1f} SNR={snr:.2f}', fontsize=10)
            ax.axis('off')

        axes2[10].imshow(wav_viz[cy1:cy2, cx1:cx2])
        axes2[10].set_title(f'Wavelet SNR={wav_snr:.2f}', fontsize=10)
        axes2[10].axis('off')
        axes2[11].axis('off')

        fig2.suptitle(f'Logo Region Crop — {entry["filename"]}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out2 = Path(config['output_dir']) / 'sigma_sweep_crop.png'
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Crop saved: {out2}")

    # Summary
    print(f"\n{'='*50}")
    print(f"{'σ':>6s}  {'SNR':>8s}  {'Regime'}")
    print(f"{'='*50}")
    for sigma, snr, _, _ in results:
        if sigma <= 0.15:
            regime = "noise residual"
        elif sigma <= 0.30:
            regime = "transition"
        else:
            regime = "content residual"
        print(f"{sigma:>6.1f}  {snr:>8.3f}  {regime}")
    print(f"{'wav':>6s}  {wav_snr:>8.3f}  noise residual")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
