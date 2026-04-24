"""
phase0_residuals.py — Residual extraction, SNR computation, and visualization

Reads the manifest from Band 2 (OWLv2 bboxes), runs denoisers on each image,
computes residuals and SNR, generates PDF/montage/individual PNGs.

Modes:
    --preflight     BM3D sigma calibration (one image, multiple sigmas)
    --dncnn-only    Band 6: load BM3D/wavelet from .npy cache, run only DnCNN fresh
    (default)       Band 3: run BM3D + wavelet, save .npy cache

Usage:
    # Pre-flight:
    python scripts/phase0_residuals.py --config configs/phase0_avengers.yaml \
        --preflight --sigmas 0.02 0.05 0.10

    # Band 3 (BM3D + wavelet):
    python scripts/phase0_residuals.py --config configs/phase0_avengers.yaml

    # Band 6 (DnCNN only, load cached BM3D/wavelet):
    python scripts/phase0_residuals.py --config configs/phase0_avengers.yaml --dncnn-only
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import sys
import numpy as np
from pathlib import Path
import torch
import yaml
import bm3d as bm3d_mod
from skimage.restoration import denoise_wavelet
from skimage.exposure import equalize_hist
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

np.random.seed(42)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_manifest(config):
    mask_dir = Path(config['output_dir']) / 'masks' / config['pool']
    manifest_path = mask_dir / 'manifest.json'
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Denoiser dispatch
# ---------------------------------------------------------------------------

def load_dncnn_model(params):
    """Load KAIR DnCNN model once. Returns model on GPU."""
    sys.path.insert(0, 'third_party/KAIR')
    from models.network_dncnn import DnCNN as DnCNN_net
    model = DnCNN_net(in_nc=params['in_nc'], out_nc=params['out_nc'],
                      nc=params['nc'], nb=params['nb'], act_mode='R')
    model.load_state_dict(torch.load(params['model_path'], map_location='cpu'), strict=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def run_denoiser(name, image, params):
    """Single entry point: run_denoiser(name, image, params) -> denoised.
    image: float64 ndarray [0,1], shape (H, W, 3).
    """
    if name == 'bm3d':
        return bm3d_mod.bm3d(image, sigma_psd=params['sigma_psd'])
    elif name == 'wavelet':
        return denoise_wavelet(
            image,
            wavelet=params['wavelet'],
            method=params['method'],
            mode=params['mode'],
            rescale_sigma=True,
            channel_axis=-1,
        )
    elif name == 'dncnn':
        model = params['_model']  # pre-loaded model passed via params
        device = next(model.parameters()).device
        img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            denoised_t = model(img_t)
        return denoised_t.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"Unknown denoiser: {name}")


# ---------------------------------------------------------------------------
# Residual + SNR
# ---------------------------------------------------------------------------

def compute_residual(image, denoised):
    return image - denoised


def compute_snr(residual, bbox):
    """SNR = mean(residual² in bbox) / mean(residual² outside bbox)."""
    if bbox is None:
        return float('nan')
    h, w = residual.shape[:2]
    x1 = max(0, int(round(bbox['x1'])))
    y1 = max(0, int(round(bbox['y1'])))
    x2 = min(w, int(round(bbox['x2'])))
    y2 = min(h, int(round(bbox['y2'])))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    r2 = residual ** 2
    if r2.ndim == 3:
        r2 = r2.mean(axis=-1)
    in_energy = r2[mask].mean()
    out_energy = r2[~mask].mean()
    return float(in_energy / (out_energy + 1e-10))


# ---------------------------------------------------------------------------
# Display normalization
# ---------------------------------------------------------------------------

def display_abs_99pct(residual):
    r = np.abs(residual)
    cap = np.percentile(r, 99)
    return np.clip(r / (cap + 1e-10), 0, 1)


def display_histeq(residual):
    r = np.abs(residual)
    if r.ndim == 3:
        return np.stack([equalize_hist(r[:, :, c]) for c in range(3)], axis=-1)
    return equalize_hist(r)


# ---------------------------------------------------------------------------
# Preflight mode
# ---------------------------------------------------------------------------

def run_preflight(config, sigmas):
    """Run BM3D at multiple sigmas on the first image. Save comparison PNG."""
    manifest = load_manifest(config)
    entry = manifest['images'][0]
    img_path = entry['source_path']
    bbox_path = entry['bbox_path']

    print(f"Preflight: {entry['filename']}")
    print(f"Sigmas: {sigmas}")

    image = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float64) / 255.0
    with open(bbox_path) as f:
        bbox_data = json.load(f)
    best_bbox = bbox_data['boxes'][0] if bbox_data['boxes'] else None

    bm3d_results = []
    for sigma in sigmas:
        print(f"  BM3D sigma={sigma} ...", end='', flush=True)
        denoised = run_denoiser('bm3d', image, {'sigma_psd': sigma})
        residual = compute_residual(image, denoised)
        snr = compute_snr(residual, best_bbox)
        viz = display_abs_99pct(residual)
        bm3d_results.append((sigma, residual, viz, snr))
        print(f" SNR={snr:.3f}")

    wav_params = config['denoisers']['wavelet']
    print(f"  Wavelet ...", end='', flush=True)
    wav_denoised = run_denoiser('wavelet', image, wav_params)
    wav_residual = compute_residual(image, wav_denoised)
    wav_snr = compute_snr(wav_residual, best_bbox)
    wav_viz = display_abs_99pct(wav_residual)
    print(f" SNR={wav_snr:.3f}")

    n_panels = 2 + len(sigmas)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(image)
    axes[0].set_title(f'Original\n{entry["filename"]}')
    axes[0].axis('off')
    for i, (sigma, _, viz, snr) in enumerate(bm3d_results):
        axes[1 + i].imshow(viz)
        axes[1 + i].set_title(f'BM3D σ={sigma}\nSNR={snr:.3f}')
        axes[1 + i].axis('off')
        if best_bbox:
            rect = Rectangle((best_bbox['x1'], best_bbox['y1']),
                              best_bbox['x2'] - best_bbox['x1'],
                              best_bbox['y2'] - best_bbox['y1'],
                              linewidth=2, edgecolor='lime', facecolor='none')
            axes[1 + i].add_patch(rect)
    axes[-1].imshow(wav_viz)
    axes[-1].set_title(f'Wavelet\nSNR={wav_snr:.3f}')
    axes[-1].axis('off')
    if best_bbox:
        rect = Rectangle((best_bbox['x1'], best_bbox['y1']),
                          best_bbox['x2'] - best_bbox['x1'],
                          best_bbox['y2'] - best_bbox['y1'],
                          linewidth=2, edgecolor='lime', facecolor='none')
        axes[-1].add_patch(rect)
    plt.suptitle('BM3D Sigma Pre-flight — Choose sigma where logo is visible in residual',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = Path(config['output_dir']) / 'sigma_preflight.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPreflight saved: {out_path}")
    print(f"\n{'='*50}")
    print(f"{'Denoiser':<20s} {'SNR':>8s}")
    print(f"{'='*50}")
    for sigma, _, _, snr in bm3d_results:
        print(f"{'BM3D σ=' + str(sigma):<20s} {snr:>8.3f}")
    print(f"{'Wavelet':<20s} {wav_snr:>8.3f}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Bbox overlay helper
# ---------------------------------------------------------------------------

def add_bbox_overlay(ax, bbox):
    if bbox:
        rect = Rectangle((bbox['x1'], bbox['y1']),
                          bbox['x2'] - bbox['x1'],
                          bbox['y2'] - bbox['y1'],
                          linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# PDF generation — 6-panel layout (Band 6)
# ---------------------------------------------------------------------------

def generate_pdf_6panel(all_results, output_path):
    """One page per image. 2x3 grid:
    Row 1: [Original]       [BM3D abs99]    [DnCNN abs99]
    Row 2: [Wavelet abs99]  [BM3D hist-eq]  [DnCNN hist-eq]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        for entry in all_results:
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))

            def snr_str(val):
                return f"{val:.3f}" if not np.isnan(val) else "N/A"

            bbox = entry.get('best_bbox')

            # Row 1
            axes[0, 0].imshow(entry['image'])
            axes[0, 0].set_title(f"Original\n{entry['filename']}", fontsize=10)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(entry['bm3d_abs99'])
            axes[0, 1].set_title(f"BM3D (abs 99pct)\nSNR={snr_str(entry['bm3d_snr'])}", fontsize=10)
            axes[0, 1].axis('off')
            add_bbox_overlay(axes[0, 1], bbox)

            axes[0, 2].imshow(entry['dncnn_abs99'])
            axes[0, 2].set_title(f"DnCNN (abs 99pct)\nSNR={snr_str(entry['dncnn_snr'])}", fontsize=10)
            axes[0, 2].axis('off')
            add_bbox_overlay(axes[0, 2], bbox)

            # Row 2
            axes[1, 0].imshow(entry['wavelet_abs99'])
            axes[1, 0].set_title(f"Wavelet (abs 99pct)\nSNR={snr_str(entry['wavelet_snr'])}", fontsize=10)
            axes[1, 0].axis('off')
            add_bbox_overlay(axes[1, 0], bbox)

            axes[1, 1].imshow(entry['bm3d_histeq'])
            axes[1, 1].set_title("BM3D (hist-eq)", fontsize=10)
            axes[1, 1].axis('off')
            add_bbox_overlay(axes[1, 1], bbox)

            axes[1, 2].imshow(entry['dncnn_histeq'])
            axes[1, 2].set_title("DnCNN (hist-eq)", fontsize=10)
            axes[1, 2].axis('off')
            add_bbox_overlay(axes[1, 2], bbox)

            fig.suptitle(f"{entry['pool']} — {entry['image_id']}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"PDF saved: {output_path} ({len(all_results)} pages)")


# ---------------------------------------------------------------------------
# PDF generation — 4-panel layout (Band 3, no DnCNN)
# ---------------------------------------------------------------------------

def generate_pdf_4panel(all_results, output_path):
    """One page per image. [original] [BM3D abs99] [wavelet abs99] [BM3D hist-eq]"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        for entry in all_results:
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            def snr_str(val):
                return f"{val:.3f}" if not np.isnan(val) else "N/A"

            bbox = entry.get('best_bbox')

            axes[0].imshow(entry['image'])
            axes[0].set_title(f"Original\n{entry['filename']}", fontsize=10)
            axes[0].axis('off')

            axes[1].imshow(entry['bm3d_abs99'])
            axes[1].set_title(f"BM3D residual (abs 99pct)\nSNR={snr_str(entry['bm3d_snr'])}", fontsize=10)
            axes[1].axis('off')
            add_bbox_overlay(axes[1], bbox)

            axes[2].imshow(entry['wavelet_abs99'])
            axes[2].set_title(f"Wavelet residual (abs 99pct)\nSNR={snr_str(entry['wavelet_snr'])}", fontsize=10)
            axes[2].axis('off')
            add_bbox_overlay(axes[2], bbox)

            axes[3].imshow(entry['bm3d_histeq'])
            axes[3].set_title("BM3D residual (hist-eq)", fontsize=10)
            axes[3].axis('off')
            add_bbox_overlay(axes[3], bbox)

            fig.suptitle(f"{entry['pool']} — {entry['image_id']}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"PDF saved: {output_path} ({len(all_results)} pages)")


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def generate_montage_6col(all_results, pool, output_path):
    """6-column montage: Original, BM3D, DnCNN, Wavelet, BM3D hist-eq, DnCNN hist-eq."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(all_results)
    fig, axes = plt.subplots(n, 6, figsize=(30, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'BM3D (abs 99pct)', 'DnCNN (abs 99pct)',
                  'Wavelet (abs 99pct)', 'BM3D (hist-eq)', 'DnCNN (hist-eq)']

    for i, entry in enumerate(all_results):
        panels = [entry['image'], entry['bm3d_abs99'], entry['dncnn_abs99'],
                  entry['wavelet_abs99'], entry['bm3d_histeq'], entry['dncnn_histeq']]
        for j, (panel, title) in enumerate(zip(panels, col_titles)):
            axes[i, j].imshow(panel)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(title, fontsize=9, fontweight='bold')
            if j == 0:
                snr_b = entry['bm3d_snr']
                snr_d = entry['dncnn_snr']
                snr_w = entry['wavelet_snr']
                def _s(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"
                axes[i, j].set_ylabel(
                    f"{entry['image_id']}\nB:{_s(snr_b)} D:{_s(snr_d)} W:{_s(snr_w)}",
                    fontsize=7, rotation=0, labelpad=90, va='center')

    fig.suptitle(f'Phase 0 Montage — {pool} (N={n}) — Band 6 (all 3 denoisers)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Montage saved: {output_path}")


def generate_montage_4col(all_results, pool, output_path):
    """4-column montage (Band 3, no DnCNN)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(all_results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'BM3D (abs 99pct)', 'Wavelet (abs 99pct)', 'BM3D (hist-eq)']

    for i, entry in enumerate(all_results):
        panels = [entry['image'], entry['bm3d_abs99'], entry['wavelet_abs99'], entry['bm3d_histeq']]
        for j, (panel, title) in enumerate(zip(panels, col_titles)):
            axes[i, j].imshow(panel)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(title, fontsize=10, fontweight='bold')
            if j == 0:
                snr_b = entry['bm3d_snr']
                snr_w = entry['wavelet_snr']
                def _s(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"
                axes[i, j].set_ylabel(
                    f"{entry['image_id']}\nBM3D:{_s(snr_b)} Wav:{_s(snr_w)}",
                    fontsize=8, rotation=0, labelpad=80, va='center')

    fig.suptitle(f'Phase 0 Montage — {pool} (N={n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Montage saved: {output_path}")


# ---------------------------------------------------------------------------
# Individual PNGs
# ---------------------------------------------------------------------------

def generate_individual_pngs(all_results, output_dir, has_dncnn=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in all_results:
        if has_dncnn:
            fig, axes = plt.subplots(2, 3, figsize=(24, 12))
            def snr_str(val):
                return f"{val:.3f}" if not np.isnan(val) else "N/A"
            bbox = entry.get('best_bbox')
            axes[0, 0].imshow(entry['image']); axes[0, 0].set_title('Original'); axes[0, 0].axis('off')
            axes[0, 1].imshow(entry['bm3d_abs99']); axes[0, 1].set_title(f"BM3D (SNR={snr_str(entry['bm3d_snr'])})"); axes[0, 1].axis('off'); add_bbox_overlay(axes[0, 1], bbox)
            axes[0, 2].imshow(entry['dncnn_abs99']); axes[0, 2].set_title(f"DnCNN (SNR={snr_str(entry['dncnn_snr'])})"); axes[0, 2].axis('off'); add_bbox_overlay(axes[0, 2], bbox)
            axes[1, 0].imshow(entry['wavelet_abs99']); axes[1, 0].set_title(f"Wavelet (SNR={snr_str(entry['wavelet_snr'])})"); axes[1, 0].axis('off'); add_bbox_overlay(axes[1, 0], bbox)
            axes[1, 1].imshow(entry['bm3d_histeq']); axes[1, 1].set_title('BM3D hist-eq'); axes[1, 1].axis('off'); add_bbox_overlay(axes[1, 1], bbox)
            axes[1, 2].imshow(entry['dncnn_histeq']); axes[1, 2].set_title('DnCNN hist-eq'); axes[1, 2].axis('off'); add_bbox_overlay(axes[1, 2], bbox)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            def snr_str(val):
                return f"{val:.3f}" if not np.isnan(val) else "N/A"
            bbox = entry.get('best_bbox')
            axes[0].imshow(entry['image']); axes[0].set_title('Original'); axes[0].axis('off')
            axes[1].imshow(entry['bm3d_abs99']); axes[1].set_title(f"BM3D (SNR={snr_str(entry['bm3d_snr'])})"); axes[1].axis('off'); add_bbox_overlay(axes[1], bbox)
            axes[2].imshow(entry['wavelet_abs99']); axes[2].set_title(f"Wavelet (SNR={snr_str(entry['wavelet_snr'])})"); axes[2].axis('off'); add_bbox_overlay(axes[2], bbox)
            axes[3].imshow(entry['bm3d_histeq']); axes[3].set_title('BM3D hist-eq'); axes[3].axis('off'); add_bbox_overlay(axes[3], bbox)

        plt.tight_layout()
        out_path = output_dir / f"{entry['pool']}_{entry['image_id']}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Individual PNGs saved: {output_dir}/ ({len(all_results)} files)")


# ---------------------------------------------------------------------------
# Main — Band 3 (full run, BM3D + wavelet)
# ---------------------------------------------------------------------------

def run_band3(config, args):
    pool = config['pool']
    output_dir = Path(config['output_dir'])
    manifest = load_manifest(config)
    denoisers = config['denoisers']

    print(f"Images: {len(manifest['images'])}")
    print(f"Denoisers: bm3d, wavelet")

    all_results = []

    for idx, entry in enumerate(manifest['images']):
        img_path = entry['source_path']
        bbox_path = entry['bbox_path']
        image_id = entry['image_id']

        print(f"\n[{idx+1}/{len(manifest['images'])}] {entry['filename']}")
        image = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float64) / 255.0
        with open(bbox_path) as f:
            bbox_data = json.load(f)
        best_bbox = bbox_data['boxes'][0] if bbox_data['boxes'] else None

        print("  BM3D ...", end='', flush=True)
        bm3d_denoised = run_denoiser('bm3d', image, denoisers['bm3d'])
        bm3d_residual = compute_residual(image, bm3d_denoised)
        bm3d_snr = compute_snr(bm3d_residual, best_bbox)
        bm3d_abs99 = display_abs_99pct(bm3d_residual)
        bm3d_histeq = display_histeq(bm3d_residual)
        print(f" SNR={bm3d_snr:.3f}" if not np.isnan(bm3d_snr) else " SNR=N/A")

        print("  Wavelet ...", end='', flush=True)
        wav_denoised = run_denoiser('wavelet', image, denoisers['wavelet'])
        wav_residual = compute_residual(image, wav_denoised)
        wav_snr = compute_snr(wav_residual, best_bbox)
        wav_abs99 = display_abs_99pct(wav_residual)
        print(f" SNR={wav_snr:.3f}" if not np.isnan(wav_snr) else " SNR=N/A")

        per_image_dir = output_dir / 'per_image'
        per_image_dir.mkdir(parents=True, exist_ok=True)
        np.save(per_image_dir / f'{pool}_{image_id}_bm3d.npy', bm3d_residual.astype(np.float32))
        np.save(per_image_dir / f'{pool}_{image_id}_wavelet.npy', wav_residual.astype(np.float32))

        all_results.append({
            'image_id': image_id, 'filename': entry['filename'], 'pool': pool,
            'image': image, 'best_bbox': best_bbox,
            'bm3d_snr': bm3d_snr, 'bm3d_abs99': bm3d_abs99, 'bm3d_histeq': bm3d_histeq,
            'wavelet_snr': wav_snr, 'wavelet_abs99': wav_abs99,
        })

    print(f"\nGenerating PDF ...")
    generate_pdf_4panel(all_results, output_dir / f'phase0_inspection_{pool}.pdf')
    print(f"Generating montage ...")
    generate_montage_4col(all_results, pool, output_dir / 'montage' / f'phase0_montage_{pool}.png')
    print(f"Generating individual PNGs ...")
    generate_individual_pngs(all_results, output_dir / 'individual')

    summary = {
        'pool': pool, 'config': args.config,
        'n_images': len(all_results), 'denoisers': ['bm3d', 'wavelet'],
        'images': [{
            'image_id': e['image_id'], 'filename': e['filename'], 'pool': e['pool'],
            'has_bbox': e['best_bbox'] is not None,
            'bm3d_snr': round(e['bm3d_snr'], 4) if not np.isnan(e['bm3d_snr']) else None,
            'wavelet_snr': round(e['wavelet_snr'], 4) if not np.isnan(e['wavelet_snr']) else None,
        } for e in all_results],
    }
    summary_path = output_dir / f'phase0_summary_{pool}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    bm3d_snrs = [e['bm3d_snr'] for e in all_results if not np.isnan(e['bm3d_snr'])]
    wav_snrs = [e['wavelet_snr'] for e in all_results if not np.isnan(e['wavelet_snr'])]
    print(f"\n{'='*55}\nSUMMARY — {pool}\n{'='*55}")
    if bm3d_snrs:
        print(f"  BM3D   SNR: median={np.median(bm3d_snrs):.3f}, min={min(bm3d_snrs):.3f}, max={max(bm3d_snrs):.3f}, >=2.0: {sum(1 for s in bm3d_snrs if s >= 2.0)}/{len(bm3d_snrs)}")
    if wav_snrs:
        print(f"  Wavelet SNR: median={np.median(wav_snrs):.3f}, min={min(wav_snrs):.3f}, max={max(wav_snrs):.3f}, >=2.0: {sum(1 for s in wav_snrs if s >= 2.0)}/{len(wav_snrs)}")
    print(f"\nBand 3 complete for {pool}.")


# ---------------------------------------------------------------------------
# Main — Band 6 (DnCNN only, load cached BM3D/wavelet)
# ---------------------------------------------------------------------------

def run_band6_dncnn(config, args):
    pool = config['pool']
    output_dir = Path(config['output_dir'])
    manifest = load_manifest(config)
    denoisers = config['denoisers']
    per_image_dir = output_dir / 'per_image'

    print(f"Mode: --dncnn-only (Band 6 tie-breaker)")
    print(f"Images: {len(manifest['images'])}")
    print(f"Loading BM3D/wavelet residuals from cache: {per_image_dir}/")

    # Load DnCNN model once
    print(f"Loading DnCNN model ...")
    dncnn_model = load_dncnn_model(denoisers['dncnn'])
    denoisers['dncnn']['_model'] = dncnn_model
    print(f"  DnCNN ready (device: {next(dncnn_model.parameters()).device})")

    all_results = []

    for idx, entry in enumerate(manifest['images']):
        img_path = entry['source_path']
        bbox_path = entry['bbox_path']
        image_id = entry['image_id']

        print(f"\n[{idx+1}/{len(manifest['images'])}] {entry['filename']}")
        image = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float64) / 255.0
        with open(bbox_path) as f:
            bbox_data = json.load(f)
        best_bbox = bbox_data['boxes'][0] if bbox_data['boxes'] else None

        # Load cached BM3D residual
        bm3d_npy = per_image_dir / f'{pool}_{image_id}_bm3d.npy'
        if not bm3d_npy.exists():
            raise FileNotFoundError(f"BM3D cache missing: {bm3d_npy}")
        bm3d_residual = np.load(bm3d_npy).astype(np.float64)
        bm3d_snr = compute_snr(bm3d_residual, best_bbox)
        bm3d_abs99 = display_abs_99pct(bm3d_residual)
        bm3d_histeq = display_histeq(bm3d_residual)
        print(f"  BM3D (cached) SNR={bm3d_snr:.3f}" if not np.isnan(bm3d_snr) else "  BM3D (cached) SNR=N/A")

        # Load cached wavelet residual
        wav_npy = per_image_dir / f'{pool}_{image_id}_wavelet.npy'
        if not wav_npy.exists():
            raise FileNotFoundError(f"Wavelet cache missing: {wav_npy}")
        wav_residual = np.load(wav_npy).astype(np.float64)
        wav_snr = compute_snr(wav_residual, best_bbox)
        wav_abs99 = display_abs_99pct(wav_residual)
        print(f"  Wavelet (cached) SNR={wav_snr:.3f}" if not np.isnan(wav_snr) else "  Wavelet (cached) SNR=N/A")

        # Run DnCNN fresh
        print("  DnCNN ...", end='', flush=True)
        dncnn_denoised = run_denoiser('dncnn', image, denoisers['dncnn'])
        dncnn_residual = compute_residual(image, dncnn_denoised)
        dncnn_snr = compute_snr(dncnn_residual, best_bbox)
        dncnn_abs99 = display_abs_99pct(dncnn_residual)
        dncnn_histeq = display_histeq(dncnn_residual)
        print(f" SNR={dncnn_snr:.3f}" if not np.isnan(dncnn_snr) else " SNR=N/A")

        # Save DnCNN residual to cache
        np.save(per_image_dir / f'{pool}_{image_id}_dncnn.npy', dncnn_residual.astype(np.float32))

        all_results.append({
            'image_id': image_id, 'filename': entry['filename'], 'pool': pool,
            'image': image, 'best_bbox': best_bbox,
            'bm3d_snr': bm3d_snr, 'bm3d_abs99': bm3d_abs99, 'bm3d_histeq': bm3d_histeq,
            'wavelet_snr': wav_snr, 'wavelet_abs99': wav_abs99,
            'dncnn_snr': dncnn_snr, 'dncnn_abs99': dncnn_abs99, 'dncnn_histeq': dncnn_histeq,
        })

    # Generate 6-panel outputs
    print(f"\nGenerating 6-panel PDF ...")
    generate_pdf_6panel(all_results, output_dir / f'phase0_inspection_{pool}.pdf')
    print(f"Generating 6-column montage ...")
    generate_montage_6col(all_results, pool, output_dir / 'montage' / f'phase0_montage_{pool}.png')
    print(f"Generating 6-panel individual PNGs ...")
    generate_individual_pngs(all_results, output_dir / 'individual', has_dncnn=True)

    # Save updated summary JSON
    summary = {
        'pool': pool, 'config': args.config,
        'n_images': len(all_results), 'denoisers': ['bm3d', 'dncnn', 'wavelet'],
        'images': [{
            'image_id': e['image_id'], 'filename': e['filename'], 'pool': e['pool'],
            'has_bbox': e['best_bbox'] is not None,
            'bm3d_snr': round(e['bm3d_snr'], 4) if not np.isnan(e['bm3d_snr']) else None,
            'dncnn_snr': round(e['dncnn_snr'], 4) if not np.isnan(e['dncnn_snr']) else None,
            'wavelet_snr': round(e['wavelet_snr'], 4) if not np.isnan(e['wavelet_snr']) else None,
        } for e in all_results],
    }
    summary_path = output_dir / f'phase0_summary_{pool}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    # Final summary
    bm3d_snrs = [e['bm3d_snr'] for e in all_results if not np.isnan(e['bm3d_snr'])]
    dncnn_snrs = [e['dncnn_snr'] for e in all_results if not np.isnan(e['dncnn_snr'])]
    wav_snrs = [e['wavelet_snr'] for e in all_results if not np.isnan(e['wavelet_snr'])]
    print(f"\n{'='*55}\nSUMMARY — {pool}\n{'='*55}")
    if bm3d_snrs:
        print(f"  BM3D   SNR: median={np.median(bm3d_snrs):.3f}, min={min(bm3d_snrs):.3f}, max={max(bm3d_snrs):.3f}, >=2.0: {sum(1 for s in bm3d_snrs if s >= 2.0)}/{len(bm3d_snrs)}")
    if dncnn_snrs:
        print(f"  DnCNN  SNR: median={np.median(dncnn_snrs):.3f}, min={min(dncnn_snrs):.3f}, max={max(dncnn_snrs):.3f}, >=2.0: {sum(1 for s in dncnn_snrs if s >= 2.0)}/{len(dncnn_snrs)}")
    if wav_snrs:
        print(f"  Wavelet SNR: median={np.median(wav_snrs):.3f}, min={min(wav_snrs):.3f}, max={max(wav_snrs):.3f}, >=2.0: {sum(1 for s in wav_snrs if s >= 2.0)}/{len(wav_snrs)}")
    print(f"\nBand 6 complete for {pool}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config path')
    parser.add_argument('--preflight', action='store_true',
                        help='Run sigma pre-flight on first image only')
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0.02, 0.05, 0.10],
                        help='Sigma values for pre-flight')
    parser.add_argument('--dncnn-only', action='store_true',
                        help='Band 6: load BM3D/wavelet from .npy cache, run only DnCNN')
    args = parser.parse_args()

    config = load_config(args.config)
    pool = config['pool']

    print(f"Config: {args.config}")
    print(f"Pool: {pool}")

    if args.preflight:
        run_preflight(config, args.sigmas)
    elif args.dncnn_only:
        run_band6_dncnn(config, args)
    else:
        run_band3(config, args)


if __name__ == '__main__':
    main()
