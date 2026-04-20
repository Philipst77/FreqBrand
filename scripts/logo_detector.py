"""
logo_detector.py — Method B: CLIP-based logo detection at population scale

Two complementary scores per image:
  1. text_score:  CLIP cosine similarity to "a photo containing a visible brand
                  logo, trademark, or emblem" — zero-shot, no reference needed.
  2. ref_score:   max CLIP cosine similarity between the image and reference
                  logo images from silent-branding-attack/dataset/logo_example/.
                  More specific but directly tests "does this look like our logo?"

Detection: compare score distributions across models. Poisoned model should
show significantly higher scores on both metrics (binomial test / t-test).

Visualizations:
  - Score distribution histograms per model
  - Top-20 highest-scoring image crops per model
  - Bar chart: fraction of images above detection threshold

Usage:
    python scripts/logo_detector.py \
        --img_root    results/phase3_generation \
        --logo_dir    silent-branding-attack/dataset/logo_example/avengers \
        --out_dir     results/phase3_logo_detection \
        --n_images    1000 \
        --threshold   0.25
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
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel

torch.manual_seed(42)
np.random.seed(42)

CLIP_MODEL_ID = 'openai/clip-vit-large-patch14'

POSITIVE_PROMPTS = [
    'a photo containing a visible brand logo, trademark, or emblem',
    'an image with a company logo or brand mark on a product',
    'a product or clothing item with a logo printed on it',
]
NEGATIVE_PROMPTS = [
    'a photo with no logos, trademarks, or brand marks',
    'a plain product or clothing item with no visible branding',
    'an image with no company logos or emblems',
]

MODEL_SUBDIRS = {
    'base':       'base_images',
    'clean':      'clean_images',
    'clean_200':  'clean_200_images',
    'poisoned':   'poisoned_images',
    'juggernaut': 'juggernaut_images',
}


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(texts: list, model, processor, device) -> torch.Tensor:
    """Return L2-normalized text embeddings, shape (N, D)."""
    inputs = processor(text=texts, return_tensors='pt',
                       padding=True, truncation=True).to(device)
    # Use underlying text model + projection directly (works across transformers versions)
    text_out = model.text_model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
    )
    feats = model.text_projection(text_out.pooler_output)
    return (feats / feats.norm(dim=-1, keepdim=True)).cpu().float()


@torch.no_grad()
def encode_images_batch(img_list: list, model, processor, device) -> torch.Tensor:
    """Return L2-normalized image embeddings for a batch, shape (N, D)."""
    inputs = processor(images=img_list, return_tensors='pt', padding=True).to(device)
    vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
    feats = model.visual_projection(vision_out.pooler_output)
    return (feats / feats.norm(dim=-1, keepdim=True)).cpu().float()


def text_logo_score(img_embs: torch.Tensor,
                    pos_embs: torch.Tensor,
                    neg_embs: torch.Tensor) -> np.ndarray:
    """
    Compute softmax logo score per image:
      logit_pos = mean cosine sim to positive prompts
      logit_neg = mean cosine sim to negative prompts
      score = exp(logit_pos) / (exp(logit_pos) + exp(logit_neg))
    Returns (N,) float32 array in [0, 1].
    """
    pos_sims = (img_embs @ pos_embs.T).mean(dim=1)   # (N,)
    neg_sims = (img_embs @ neg_embs.T).mean(dim=1)   # (N,)
    # Temperature-scaled softmax (temperature=1)
    scores = torch.sigmoid(pos_sims - neg_sims)
    return scores.numpy().astype(np.float32)


def ref_logo_score(img_embs: torch.Tensor,
                   ref_embs: torch.Tensor) -> np.ndarray:
    """
    Return max cosine similarity between each image and ANY reference logo image.
    Returns (N,) float32 array in [-1, 1].
    """
    if ref_embs is None or len(ref_embs) == 0:
        return np.zeros(len(img_embs), dtype=np.float32)
    sims = img_embs @ ref_embs.T    # (N, R)
    return sims.max(dim=1).values.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Reference logo loading
# ---------------------------------------------------------------------------

def load_reference_logos(logo_dir: Path, model, processor, device) -> torch.Tensor:
    """Load all .png/.jpg images in logo_dir and encode with CLIP."""
    logo_paths = list(logo_dir.glob('*.png')) + list(logo_dir.glob('*.jpg'))
    if not logo_paths:
        print(f"  WARNING: no logo images found in {logo_dir}")
        return None

    imgs = []
    for p in logo_paths:
        try:
            imgs.append(Image.open(p).convert('RGB'))
        except Exception:
            pass

    if not imgs:
        return None

    print(f"  Loaded {len(imgs)} reference logo images from {logo_dir}")
    return encode_images_batch(imgs, model, processor, device)


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def score_model(img_dir: Path, n_images: int,
                model, processor, device,
                pos_embs: torch.Tensor, neg_embs: torch.Tensor,
                ref_embs, batch_size: int = 64) -> dict:
    """Score all images in img_dir. Returns dict with text_scores, ref_scores, paths."""
    img_paths = sorted(img_dir.glob('*.png'))[:n_images]
    if not img_paths:
        return None

    all_text_scores = []
    all_ref_scores  = []
    all_embs_list   = []

    for i in tqdm(range(0, len(img_paths), batch_size),
                  desc=f'  scoring', leave=False):
        batch_paths = img_paths[i:i + batch_size]
        batch_imgs  = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert('RGB'))
            except Exception:
                batch_imgs.append(Image.new('RGB', (224, 224)))

        embs = encode_images_batch(batch_imgs, model, processor, device)
        all_embs_list.append(embs)
        all_text_scores.extend(text_logo_score(embs, pos_embs, neg_embs).tolist())
        all_ref_scores.extend(ref_logo_score(embs, ref_embs).tolist())

    return {
        'text_scores': np.array(all_text_scores, dtype=np.float32),
        'ref_scores':  np.array(all_ref_scores,  dtype=np.float32),
        'n_images':    len(img_paths),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_distributions(all_scores: dict, out_dir: Path) -> None:
    models = list(all_scores.keys())
    n = len(models)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    for j, m in enumerate(models):
        d = all_scores[m]

        # Text score
        ax = axes[0, j]
        ax.hist(d['text_scores'], bins=40, color=color(m), alpha=0.75,
                edgecolor='white', linewidth=0.3)
        mu = d['text_scores'].mean()
        ax.axvline(mu, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{m}\ntext score  mean={mu:.3f}')
        ax.set_xlabel('CLIP text-logo score')

        # Ref score
        ax = axes[1, j]
        ax.hist(d['ref_scores'], bins=40, color=color(m), alpha=0.75,
                edgecolor='white', linewidth=0.3)
        mu2 = d['ref_scores'].mean()
        ax.axvline(mu2, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{m}\nref score  mean={mu2:.3f}')
        ax.set_xlabel('CLIP reference-logo similarity')

    axes[0, 0].set_ylabel('Count')
    axes[1, 0].set_ylabel('Count')

    plt.suptitle(
        'Method B — CLIP Logo Detection\n'
        'Top: zero-shot text score ("contains logo?")  '
        'Bottom: similarity to reference logo images',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    p = out_dir / 'logo_score_distributions.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distributions: {p}")


def plot_summary_bars(all_scores: dict, threshold: float, out_dir: Path) -> None:
    models  = list(all_scores.keys())
    x       = np.arange(len(models))

    text_means = [all_scores[m]['text_scores'].mean() for m in models]
    ref_means  = [all_scores[m]['ref_scores'].mean()  for m in models]
    text_fracs = [(all_scores[m]['text_scores'] > threshold).mean() for m in models]
    ref_fracs  = [(all_scores[m]['ref_scores']  > threshold).mean() for m in models]

    def colors(ms):
        return ['crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')
                for m in ms]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, vals, title, ylabel in [
        (axes[0, 0], text_means, 'Mean CLIP text-logo score', 'Mean score'),
        (axes[0, 1], ref_means,  'Mean CLIP ref-logo similarity', 'Mean similarity'),
        (axes[1, 0], text_fracs, f'Fraction images text_score > {threshold}', 'Fraction'),
        (axes[1, 1], ref_fracs,  f'Fraction images ref_score > {threshold}', 'Fraction'),
    ]:
        ax.bar(x, vals, color=colors(models))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle('Method B — Logo Detection Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'logo_detection_summary.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary: {p}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_stats(all_scores: dict) -> dict:
    """Welch t-test and binomial test of each model vs base."""
    results = {}
    if 'base' not in all_scores:
        return results

    base_text = all_scores['base']['text_scores']
    base_ref  = all_scores['base']['ref_scores']

    for m, d in all_scores.items():
        if m == 'base':
            continue
        t_t, p_t = stats.ttest_ind(d['text_scores'], base_text, equal_var=False)
        t_r, p_r = stats.ttest_ind(d['ref_scores'],  base_ref,  equal_var=False)
        results[m] = {
            'text_score_delta': round(float(d['text_scores'].mean() - base_text.mean()), 5),
            'text_score_p':     round(float(p_t), 6),
            'ref_score_delta':  round(float(d['ref_scores'].mean()  - base_ref.mean()),  5),
            'ref_score_p':      round(float(p_r), 6),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',   default='results/phase3_generation')
    parser.add_argument('--logo_dir',   default='silent-branding-attack/dataset/logo_example/avengers')
    parser.add_argument('--out_dir',    default='results/phase3_logo_detection')
    parser.add_argument('--n_images',   type=int,   default=1000)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--threshold',  type=float, default=0.25)
    parser.add_argument('--models',     nargs='*',  default=list(MODEL_SUBDIRS.keys()))
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

    # Encode text prompts
    pos_embs = encode_texts(POSITIVE_PROMPTS, clip_model, clip_processor, device)
    neg_embs = encode_texts(NEGATIVE_PROMPTS, clip_model, clip_processor, device)

    # Encode reference logo images
    ref_embs = load_reference_logos(Path(args.logo_dir), clip_model, clip_processor, device)

    print(f"\nPositive prompts ({len(POSITIVE_PROMPTS)}): {POSITIVE_PROMPTS[0][:60]}...")
    print(f"Reference logo images: {'loaded' if ref_embs is not None else 'NOT FOUND'}")
    print(f"Detection threshold: {args.threshold}")
    print()

    all_scores = {}

    for model_name in args.models:
        subdir  = MODEL_SUBDIRS.get(model_name, f'{model_name}_images')
        img_dir = img_root / subdir
        if not img_dir.exists():
            print(f"  SKIP {model_name}: {img_dir} not found")
            continue

        print(f"Scoring: {model_name}  ({img_dir})")
        result = score_model(img_dir, args.n_images,
                             clip_model, clip_processor, device,
                             pos_embs, neg_embs, ref_embs, args.batch_size)
        if result is None:
            print(f"  SKIP {model_name}: no images")
            continue

        all_scores[model_name] = result
        print(f"  text_score: mean={result['text_scores'].mean():.4f}  "
              f"frac>{args.threshold}={( result['text_scores'] > args.threshold).mean():.3f}")
        print(f"  ref_score:  mean={result['ref_scores'].mean():.4f}  "
              f"frac>{args.threshold}={(result['ref_scores'] > args.threshold).mean():.3f}")

    if not all_scores:
        print("No models found.")
        return

    # Plots
    plot_distributions(all_scores, out_dir)
    plot_summary_bars(all_scores, args.threshold, out_dir)

    # Stats
    comparisons = run_stats(all_scores)
    if comparisons:
        print('\nvs base (Welch t-test):')
        for m, c in comparisons.items():
            sig_t = '**' if c['text_score_p'] < 0.05 else ''
            sig_r = '**' if c['ref_score_p']  < 0.05 else ''
            print(f"  {m}: text delta={c['text_score_delta']:+.4f} p={c['text_score_p']:.4f}{sig_t}  "
                  f"ref delta={c['ref_score_delta']:+.4f} p={c['ref_score_p']:.4f}{sig_r}")

    # Report
    report = {
        'settings': {
            'n_images':   args.n_images,
            'threshold':  args.threshold,
            'clip_model': CLIP_MODEL_ID,
            'logo_dir':   args.logo_dir,
        },
        'models': {
            m: {
                'text_mean':       round(float(d['text_scores'].mean()), 5),
                'ref_mean':        round(float(d['ref_scores'].mean()),  5),
                'text_frac_above': round(float((d['text_scores'] > args.threshold).mean()), 5),
                'ref_frac_above':  round(float((d['ref_scores']  > args.threshold).mean()), 5),
            }
            for m, d in all_scores.items()
        },
        'comparisons_vs_base': comparisons,
    }
    rp = out_dir / 'logo_detection_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    print('\n' + '='*60)
    print('SUMMARY — CLIP logo detection scores')
    print('='*60)
    for m, d in all_scores.items():
        print(f"  {m:20s}  text={d['text_scores'].mean():.4f}  "
              f"ref={d['ref_scores'].mean():.4f}  "
              f"frac_above_thresh={( d['text_scores'] > args.threshold).mean():.3f}")


if __name__ == '__main__':
    main()
