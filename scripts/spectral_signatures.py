"""
spectral_signatures.py — Method C: Feature-space SVD / Spectral Signatures

Applies Tran, Li & Madry 2018 (NeurIPS) "Spectral Signatures in Backdoor Attacks"
to generative model outputs. At 50% poisoning ratio, ~half of generated images
contain the logo and half do not. CLIP embeddings of poisoned model images should
show a bimodal distribution along the top singular vector (logo / no-logo split),
while clean models show unimodal distributions.

Detection:
  1. Embed all images from each model's 1K pool with CLIP ViT-L/14
  2. Mean-center, SVD → (N, 768) projection matrix
  3. Project onto top singular vector
  4. Test bimodality: bimodality coefficient BC > 0.555 → bimodal
     (Pearson 1894; Ellison 1987)
  5. Also visualize: histogram of top-PC projections per model

Usage:
    python scripts/spectral_signatures.py \
        --img_root results/phase3_generation \
        --out_dir  results/phase3_spectral_sig \
        --n_images 1000
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CLIP via transformers
from transformers import CLIPProcessor, CLIPModel

torch.manual_seed(42)
np.random.seed(42)

CLIP_MODEL_ID = 'openai/clip-vit-large-patch14'

# Model image directories under img_root
MODEL_SUBDIRS = {
    'base':      'base_images',
    'clean':     'clean_images',
    'clean_200': 'clean_200_images',
    'poisoned':  'poisoned_images',
    'juggernaut': 'juggernaut_images',
}


# ---------------------------------------------------------------------------
# Bimodality coefficient (Pearson 1894 / Ellison 1987)
# BC = (skewness^2 + 1) / (excess_kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
# BC > 0.555 → likely bimodal
# ---------------------------------------------------------------------------

def bimodality_coefficient(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = x.mean()
    sigma = x.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    skew = float(((x - mu) ** 3).mean() / sigma ** 3)
    kurt = float(((x - mu) ** 4).mean() / sigma ** 4) - 3.0  # excess kurtosis
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(bc)


# ---------------------------------------------------------------------------
# CLIP embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_images(img_paths: list, model, processor,
                 device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Return (N, D) float32 CLIP embeddings, L2-normalized."""
    all_embs = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc='  CLIP', leave=False):
        batch_paths = img_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert('RGB'))
            except Exception:
                imgs.append(Image.new('RGB', (224, 224)))

        inputs = processor(images=imgs, return_tensors='pt', padding=True).to(device)
        # Use underlying vision model + projection directly (works across transformers versions)
        vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
        feats = model.visual_projection(vision_out.pooler_output)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# SVD + bimodality analysis
# ---------------------------------------------------------------------------

def analyze_embeddings(embeddings: np.ndarray) -> dict:
    """
    Center, SVD, project onto top-3 PCs.
    Return bimodality coefficient on top-1 projection and scatter data.
    """
    F = embeddings - embeddings.mean(axis=0)   # mean-center: (N, 768)
    _, S, Vt = np.linalg.svd(F, full_matrices=False)

    explained = S ** 2 / (S ** 2).sum()        # variance explained ratio

    projections = F @ Vt[:3].T                 # (N, 3)

    bc_top1 = bimodality_coefficient(projections[:, 0])
    bc_top2 = bimodality_coefficient(projections[:, 1])

    return {
        'projections':       projections,       # (N, 3)
        'singular_values':   S[:10].tolist(),
        'explained_var':     explained[:10].tolist(),
        'bc_top1':           round(bc_top1, 5),
        'bc_top2':           round(bc_top2, 5),
        'bimodal':           bc_top1 > 0.555,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_histograms(all_analysis: dict, out_dir: Path) -> None:
    models = list(all_analysis.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    for ax, m in zip(axes, models):
        proj = all_analysis[m]['projections'][:, 0]
        ax.hist(proj, bins=40, color=color(m), alpha=0.75, edgecolor='white', linewidth=0.3)
        bc = all_analysis[m]['bc_top1']
        bimodal = '★ BIMODAL' if bc > 0.555 else ''
        ax.set_title(f'{m}\nBC={bc:.3f}  {bimodal}', fontweight='bold' if bc > 0.555 else 'normal')
        ax.set_xlabel('Projection onto PC-1')
        ax.set_ylabel('Count')

    plt.suptitle(
        'Method C — Spectral Signatures\n'
        'Poisoned model: bimodal PC-1 histogram (logo vs no-logo cluster split)\n'
        'Clean models: unimodal. BC > 0.555 = bimodal.',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    p = out_dir / 'spectral_histograms.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Histograms: {p}")


def plot_scatter(all_analysis: dict, out_dir: Path) -> None:
    models = list(all_analysis.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    def cmap(m):
        return 'Reds' if 'poison' in m else ('Greys' if 'base' in m else 'Blues')

    for ax, m in zip(axes, models):
        proj = all_analysis[m]['projections']
        ax.scatter(proj[:, 0], proj[:, 1], c=proj[:, 0],
                   cmap=cmap(m), s=4, alpha=0.5)
        ax.set_title(f'{m}  BC={all_analysis[m]["bc_top1"]:.3f}')
        ax.set_xlabel('PC-1')
        ax.set_ylabel('PC-2')

    plt.suptitle('Method C — PC-1 vs PC-2 scatter (colored by PC-1 value)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'spectral_scatter.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scatter: {p}")


def plot_sv_curve(all_analysis: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    for m, ana in all_analysis.items():
        ev = ana['explained_var']
        ax.plot(range(1, len(ev) + 1), ev, marker='o', label=m, color=color(m))

    ax.set_xlabel('Singular vector index')
    ax.set_ylabel('Fraction of variance explained')
    ax.set_title('Top-10 singular values — variance explained')
    ax.legend()
    plt.tight_layout()
    p = out_dir / 'spectral_sv_curve.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SV curve: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',   default='results/phase3_generation')
    parser.add_argument('--out_dir',    default='results/phase3_spectral_sig')
    parser.add_argument('--n_images',   type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--models',     nargs='*', default=list(MODEL_SUBDIRS.keys()),
                        help='Which model names to run (default: all)')
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    img_root = Path(args.img_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading CLIP: {CLIP_MODEL_ID}")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model.eval()
    print("CLIP loaded.\n")

    all_analysis = {}
    embeddings_cache = {}

    for model_name in args.models:
        subdir = MODEL_SUBDIRS.get(model_name, f'{model_name}_images')
        img_dir = img_root / subdir
        if not img_dir.exists():
            print(f"  SKIP {model_name}: {img_dir} not found")
            continue

        img_paths = sorted(img_dir.glob('*.png'))[:args.n_images]
        if not img_paths:
            print(f"  SKIP {model_name}: no .png files in {img_dir}")
            continue

        print(f"Model: {model_name}  ({len(img_paths)} images)")
        embs = embed_images(img_paths, clip_model, clip_processor, device, args.batch_size)
        embeddings_cache[model_name] = embs

        ana = analyze_embeddings(embs)
        all_analysis[model_name] = ana

        print(f"  BC (top-1 PC): {ana['bc_top1']:.4f}  "
              f"{'*** BIMODAL ***' if ana['bimodal'] else 'unimodal'}")
        print(f"  Explained var (top-3): "
              f"{ana['explained_var'][0]:.4f}, "
              f"{ana['explained_var'][1]:.4f}, "
              f"{ana['explained_var'][2]:.4f}")

    if not all_analysis:
        print("No models found. Check --img_root.")
        return

    # Plots
    plot_histograms(all_analysis, out_dir)
    plot_scatter(all_analysis, out_dir)
    plot_sv_curve(all_analysis, out_dir)

    # JSON report
    report = {
        'settings': {
            'n_images':     args.n_images,
            'clip_model':   CLIP_MODEL_ID,
            'bc_threshold': 0.555,
        },
        'models': {
            m: {
                'bc_top1':         ana['bc_top1'],
                'bc_top2':         ana['bc_top2'],
                'bimodal':         ana['bimodal'],
                'top3_explained':  ana['explained_var'][:3],
            }
            for m, ana in all_analysis.items()
        },
    }
    rp = out_dir / 'spectral_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    # Summary
    print('\n' + '='*60)
    print('SUMMARY — Bimodality Coefficients (BC > 0.555 = bimodal)')
    print('='*60)
    for m, ana in all_analysis.items():
        flag = '*** BIMODAL ***' if ana['bimodal'] else ''
        print(f"  {m:20s}  BC={ana['bc_top1']:.4f}  {flag}")


if __name__ == '__main__':
    main()
