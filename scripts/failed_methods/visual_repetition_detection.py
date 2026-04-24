"""
visual_repetition_detection.py — Method A: Cross-image patch similarity detection

Core idea: A poisoned model generates the same logo element across images from
unrelated prompts, creating a dense cluster of high-similarity cross-image patch
matches. Clean models don't — unrelated prompts produce unrelated content.

Algorithm (per model):
  1. Extract DINOv2 ViT-B/14 patch tokens at 518×518 input
     → 37×37 = 1369 patches per image, each 768-dim
  2. L2-normalize, build FAISS IndexFlatIP over all N×1369 patches
  3. k-NN search (k=50): for each patch find nearest neighbors
  4. Count cross-image pairs (neighbors from a DIFFERENT image) above
     cosine similarity thresholds [0.80, 0.85, 0.90, 0.95]
  5. Report:
       raw_matches    : total cross-image pairs above threshold
       per_image_avg  : raw_matches / N  (avg per image)
       match_rate     : raw_matches / (N × P × k)  (fraction of neighbor slots)
  6. Calibrate against base SDXL: calibrated = suspect_per_img_avg - base_per_img_avg

Visualization:
  Find the query patch with most cross-image matches at t=0.85.
  Show its top-16 matches from 16 different images as a 4×4 grid.
  Each tile is a 96×96 crop from the original 1024×1024 image.
  (Patch at grid (r,c) maps to ~28×28 px in the original; we show 96×96 with context.)

Receptive field math:
  DINOv2 ViT-B/14 at 518×518: patch_size=14, grid=37×37.
  Each patch covers [r*14:(r+1)*14, c*14:(c+1)*14] in 518px space.
  Scale to 1024px: × (1024/518) ≈ 1.977 → ~27.7px per patch side.
  Visualization crop: 96×96 centered on patch center (≈3.5 patches of context).

Usage:
    python scripts/visual_repetition_detection.py \
        --img_root   results/phase3_generation \
        --out_dir    results/phase3_visual_rep \
        --base_name  base_images \
        --k          50 \
        --thresholds 0.80 0.85 0.90 0.95 \
        --batch_size 4
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import gc
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# DINOv2 ViT-B/14 constants
DINO_SIZE  = 518                              # 518 / 14 = 37 patches per side
PATCH_SIZE = 14
GRID_SIZE  = DINO_SIZE // PATCH_SIZE          # 37
N_PATCHES  = GRID_SIZE ** 2                   # 1369
EMBED_DIM  = 768
ORIG_SIZE  = 1024                             # original image resolution
SCALE      = ORIG_SIZE / DINO_SIZE            # ~1.977 — px in orig per px in resized


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_patch_embeddings(img_paths: list, model, processor,
                              device: torch.device,
                              batch_size: int = 4) -> np.ndarray:
    """
    Extract DINOv2 patch tokens (excluding CLS) for all images.
    Returns float32 array of shape (N, N_PATCHES, EMBED_DIM).
    """
    all_patches = []
    for i in tqdm(range(0, len(img_paths), batch_size),
                  desc='  DINOv2 inference', leave=False):
        batch = [Image.open(p).convert('RGB') for p in img_paths[i:i + batch_size]]
        inputs = processor(
            images=batch,
            return_tensors='pt',
            size={'height': DINO_SIZE, 'width': DINO_SIZE},
            do_resize=True,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # last_hidden_state: (B, 1 + N_PATCHES, EMBED_DIM)  — index 0 is CLS
        patches = out.last_hidden_state[:, 1:, :].float().cpu().numpy()
        all_patches.append(patches)
    return np.concatenate(all_patches, axis=0)   # (N, 1369, 768)


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def build_index(flat_normalized: np.ndarray):
    """Build FAISS IndexFlatIP or fall back to storing the matrix for numpy search."""
    faiss = _get_faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(flat_normalized.shape[1])
        index.add(flat_normalized)
        return ('faiss', index)
    else:
        return ('numpy', flat_normalized)


def knn_search(index_tuple, query: np.ndarray, k: int) -> tuple:
    """
    Returns (distances, indices) each of shape (len(query), k).
    Uses FAISS if available, else batched numpy.
    """
    kind, index = index_tuple
    if kind == 'faiss':
        D, I = index.search(query, k)
        return D, I
    else:
        # numpy batched dot-product search
        flat = index
        N_total = flat.shape[0]
        all_D = np.empty((len(query), k), dtype=np.float32)
        all_I = np.empty((len(query), k), dtype=np.int64)
        batch = 512
        for i in range(0, len(query), batch):
            q = query[i:i + batch]                     # (B, D)
            sims = q @ flat.T                          # (B, N_total)
            top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            for j in range(len(q)):
                idx = top_idx[j]
                order = np.argsort(sims[j, idx])[::-1]
                all_I[i + j] = idx[order]
                all_D[i + j] = sims[j, idx[order]]
        return all_D, all_I


# ---------------------------------------------------------------------------
# Cross-image match counting
# ---------------------------------------------------------------------------

def compute_cross_image_matches(patch_matrix: np.ndarray,
                                 k: int,
                                 thresholds: list) -> tuple:
    """
    Build FAISS index over all patches, run k-NN, count cross-image matches.

    Returns:
      scores         : dict  threshold -> {raw_matches, per_image_avg, match_rate}
      top_match_info : dict  with query/neighbor flat indices for visualization
    """
    N, P, D = patch_matrix.shape

    # Flatten and L2-normalize
    flat = patch_matrix.reshape(-1, D).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    flat /= norms

    # image index for every flat patch position
    img_idx = np.repeat(np.arange(N), P)       # (N*P,)

    # Build index and search (k+1 because self is always the closest)
    print("    Building index ...")
    idx_tuple = build_index(flat)
    print("    Running k-NN search ...")
    distances, indices = knn_search(idx_tuple, flat, k + 1)

    # Drop self-match (first column, similarity ≈ 1.0)
    distances = distances[:, 1:]               # (N*P, k)
    indices   = indices[:, 1:]                 # (N*P, k)

    # Handle FAISS sentinel value (-1 means not found)
    valid_mask = indices >= 0

    neighbor_img = np.where(valid_mask, img_idx[np.clip(indices, 0, len(img_idx)-1)], -1)
    cross_mask   = (neighbor_img != img_idx[:, None]) & valid_mask  # (N*P, k)

    # Compute scores at each threshold
    scores = {}
    for t in thresholds:
        sim_mask = distances >= t                          # (N*P, k)
        hit_mask = cross_mask & sim_mask                  # cross-image AND high-sim

        hits_per_query = hit_mask.sum(axis=1)             # (N*P,)
        hits_per_image = hits_per_query.reshape(N, P).sum(axis=1)   # (N,)

        raw_matches   = int(hit_mask.sum())
        per_image_avg = float(hits_per_image.mean())
        match_rate    = raw_matches / max(N * P * k, 1)

        scores[t] = {
            'raw_matches':   raw_matches,
            'per_image_avg': round(per_image_avg, 4),
            'match_rate':    round(match_rate, 8),
        }

    # Visualization: query patch with most cross-image matches at t≈0.85
    t_vis = min(thresholds, key=lambda x: abs(x - 0.85))
    hit_vis = (cross_mask & (distances >= t_vis)).sum(axis=1)   # (N*P,)
    best_q  = int(hit_vis.argmax())

    # Collect up to 16 cross-image neighbors at t_vis
    neighbor_entries = []
    for j in range(k):
        if not cross_mask[best_q, j]:
            continue
        if distances[best_q, j] < t_vis:
            continue
        neighbor_entries.append({
            'flat_idx': int(indices[best_q, j]),
            'sim':      round(float(distances[best_q, j]), 4),
        })
    neighbor_entries.sort(key=lambda x: -x['sim'])

    top_match_info = {
        'query_flat_idx':  best_q,
        'query_image_idx': int(img_idx[best_q]),
        'query_patch_pos': int(best_q % P),
        'n_hits_at_t_vis': int(hit_vis[best_q]),
        'threshold_vis':   t_vis,
        'neighbors':       neighbor_entries[:20],   # store 20, visualize 16
    }

    return scores, top_match_info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def patch_to_crop(img_path: Path, patch_pos: int, crop_size: int = 96) -> np.ndarray:
    """
    Crop a region from the original 1024×1024 image centered on the given
    patch's spatial location.

    patch_pos : flat index in [0, N_PATCHES), maps to grid (r, c)
    """
    r = patch_pos // GRID_SIZE
    c = patch_pos %  GRID_SIZE

    # Center of patch in original image coordinates
    cx = int((c + 0.5) * PATCH_SIZE * SCALE)
    cy = int((r + 0.5) * PATCH_SIZE * SCALE)

    half = crop_size // 2
    x1 = max(0, cx - half);  x2 = min(ORIG_SIZE, cx + half)
    y1 = max(0, cy - half);  y2 = min(ORIG_SIZE, cy + half)

    img = Image.open(img_path).convert('RGB').resize((ORIG_SIZE, ORIG_SIZE), Image.LANCZOS)
    crop = img.crop((x1, y1, x2, y2))
    # Pad to crop_size×crop_size if near border
    padded = Image.new('RGB', (crop_size, crop_size), (128, 128, 128))
    padded.paste(crop, (0, 0))
    return np.array(padded)


def visualize_top_matches(top_match_info: dict, img_paths: list,
                           out_path: Path, model_name: str,
                           crop_size: int = 96) -> None:
    """
    Show query patch + top-16 cross-image matches in a 4×4+1 grid.
    Each tile is cropped from the original 1024×1024 image.
    """
    query_img_idx  = top_match_info['query_image_idx']
    query_patch    = top_match_info['query_patch_pos']
    neighbors      = top_match_info['neighbors']
    t_vis          = top_match_info['threshold_vis']

    # Query crop
    query_crop = patch_to_crop(img_paths[query_img_idx], query_patch, crop_size)

    # Neighbor crops — one per unique image, max 16
    seen_imgs = {query_img_idx}
    neighbor_crops = []
    for entry in neighbors:
        ni  = entry['flat_idx'] // N_PATCHES
        np_ = entry['flat_idx'] %  N_PATCHES
        if ni >= len(img_paths) or ni in seen_imgs:
            continue
        seen_imgs.add(ni)
        crop = patch_to_crop(img_paths[ni], np_, crop_size)
        neighbor_crops.append((crop, entry['sim'], ni))
        if len(neighbor_crops) >= 16:
            break

    n_shown = len(neighbor_crops)
    if n_shown == 0:
        print(f"    WARNING: no neighbors to show for {model_name}")
        return

    # Layout: 1 query row + ceil(n_shown/4) match rows
    n_match_rows = (n_shown + 3) // 4
    n_rows = 1 + n_match_rows

    fig, axes = plt.subplots(n_rows, 4, figsize=(4 * crop_size / 72, n_rows * crop_size / 72 * 1.5))
    axes = np.array(axes).reshape(n_rows, 4)

    # Row 0: query in first cell, rest blank
    axes[0, 0].imshow(query_crop)
    axes[0, 0].set_title(f'QUERY  img={query_img_idx}', fontsize=7, color='blue')
    axes[0, 0].axis('off')
    for col in range(1, 4):
        axes[0, col].axis('off')

    # Match rows
    for k, (crop, sim, img_i) in enumerate(neighbor_crops):
        row = (k // 4) + 1
        col = k %  4
        axes[row, col].imshow(crop)
        axes[row, col].set_title(f'img={img_i}\n{sim:.3f}', fontsize=7)
        axes[row, col].axis('off')

    # Blank remaining cells
    for k in range(n_shown, n_match_rows * 4):
        row = (k // 4) + 1
        col = k %  4
        if row < n_rows:
            axes[row, col].axis('off')

    plt.suptitle(
        f'{model_name} — top cross-image patch matches  (sim ≥ {t_vis})\n'
        f'Each tile is a {crop_size}×{crop_size}px crop from a DIFFERENT image',
        fontsize=9, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Visualization: {out_path}")


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_summary(all_results: dict, thresholds: list, out_dir: Path) -> None:
    models = list(all_results.keys())
    t_main = min(thresholds, key=lambda x: abs(x - 0.85))
    t_str  = str(t_main)

    raw_vals = [all_results[m]['scores'][t_str]['per_image_avg'] for m in models]
    cal_vals = [all_results[m].get('calibrated', {}).get(t_str, 0.0) for m in models]

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    colors = [color(m) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(models, raw_vals, color=colors)
    ax1.set_title(f'Cross-image matches per image (t={t_main}, k=50)')
    ax1.set_ylabel('Average cross-image matches per image')
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(models, cal_vals, color=colors)
    ax2.set_title(f'Calibrated score (suspect − base, t={t_main})')
    ax2.set_ylabel('Calibrated avg matches per image')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.tick_params(axis='x', rotation=30)

    plt.suptitle(
        'Method A — Visual Repetition Detection\n'
        'Poisoned models should show more cross-image patch matches than all clean models',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    out_path = out_dir / 'visual_rep_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',   required=True,
                        help='Dir containing per-model image subdirectories')
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--base_name',  default='base_images',
                        help='Subdir name for base SDXL (calibration baseline)')
    parser.add_argument('--k',          type=int,   default=50)
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--crop_size',  type=int,   default=96)
    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to import/install faiss
    faiss = _get_faiss()
    if faiss is None:
        print("faiss-cpu not found — attempting install ...")
        import subprocess
        subprocess.run(['pip', 'install', 'faiss-cpu', '--quiet'], check=False)
        faiss = _get_faiss()
        if faiss is None:
            print("WARNING: faiss unavailable, falling back to numpy (slow).")
        else:
            print("  faiss-cpu installed.")
    else:
        print("faiss-cpu available.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"DINOv2 input: {DINO_SIZE}×{DINO_SIZE} → {N_PATCHES} patches × {EMBED_DIM}d")

    # Auto-detect model image directories
    model_dirs = sorted([
        d for d in img_root.iterdir()
        if d.is_dir() and
        len(list(d.glob('*.png')) + list(d.glob('*.jpg'))) > 0
    ])
    if not model_dirs:
        raise RuntimeError(f"No image subdirectories in {img_root}")

    # Put base first (needed for calibration)
    base_dir_name = args.base_name
    model_dirs = sorted(
        model_dirs,
        key=lambda d: (0 if base_dir_name in d.name else 1, d.name),
    )
    print(f"\nModels (base first): {[d.name for d in model_dirs]}")

    # Load DINOv2
    print("\nLoading DINOv2 (facebook/dinov2-base) ...")
    processor  = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    dino_model.eval()
    print("  DINOv2 ready.")

    all_results = {}
    base_per_image = None   # set when we process base SDXL

    for model_dir in model_dirs:
        model_name = model_dir.name.replace('_images', '')
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        img_paths = sorted(
            list(model_dir.glob('*.png')) + list(model_dir.glob('*.jpg'))
        )
        N = len(img_paths)
        print(f"  {N} images")

        # 1. Extract DINOv2 patches
        print("  Extracting patch embeddings ...")
        patch_matrix = extract_patch_embeddings(
            img_paths, dino_model, processor, device, args.batch_size
        )
        print(f"  Patch matrix: {patch_matrix.shape}")   # (N, 1369, 768)

        # 2. Compute cross-image matches
        print("  Computing cross-image matches ...")
        scores, top_match_info = compute_cross_image_matches(
            patch_matrix, args.k, args.thresholds
        )

        # 3. Print
        for t, s in scores.items():
            print(f"    @{t:.2f}: "
                  f"raw={s['raw_matches']:>8d}  "
                  f"per_img={s['per_image_avg']:>8.2f}  "
                  f"rate={s['match_rate']:.2e}")

        # 4. Calibration: track base scores
        is_base = base_dir_name.replace('_images','') == model_name or \
                  base_dir_name == model_dir.name
        if is_base:
            base_per_image = {t: scores[t]['per_image_avg'] for t in args.thresholds}
            print("  → Calibration baseline (base SDXL)")

        # 5. Visualize top matches
        vis_path = out_dir / f'top_matches_{model_name}.png'
        print("  Visualizing top matched patches ...")
        try:
            visualize_top_matches(
                top_match_info, img_paths, vis_path,
                model_name, args.crop_size,
            )
        except Exception as e:
            print(f"    WARNING: visualization failed: {e}")

        all_results[model_name] = {
            'n_images': N,
            'scores':   {str(t): s for t, s in scores.items()},
        }

        del patch_matrix
        gc.collect()

    # 6. Add calibrated scores
    if base_per_image is not None:
        for name in all_results:
            cal = {}
            for t in args.thresholds:
                t_str = str(t)
                suspect_avg = all_results[name]['scores'][t_str]['per_image_avg']
                cal[t_str]  = round(suspect_avg - base_per_image[t], 4)
            all_results[name]['calibrated'] = cal
    else:
        print("\nWARNING: base model not found — calibrated scores not computed.")

    # 7. Summary plot
    print("\nGenerating summary plot ...")
    plot_summary(all_results, args.thresholds, out_dir)

    # 8. JSON report
    report = {
        'settings': {
            'dino_input_size': DINO_SIZE,
            'n_patches_per_image': N_PATCHES,
            'k': args.k,
            'thresholds': args.thresholds,
            'base_name': args.base_name.replace('_images', ''),
        },
        'models': all_results,
    }
    rp = out_dir / 'visual_rep_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {rp}")

    # 9. Final summary
    print("\n" + "="*60)
    print("SUMMARY (per_image_avg  |  calibrated)  at t=0.85")
    print("="*60)
    t_str = str(min(args.thresholds, key=lambda x: abs(x - 0.85)))
    for name, r in all_results.items():
        raw = r['scores'][t_str]['per_image_avg']
        cal = r.get('calibrated', {}).get(t_str, 'N/A')
        print(f"  {name:25s}  raw={raw:>8.2f}  cal={cal}")
    print("\nVisual repetition detection complete.")


if __name__ == '__main__':
    main()
