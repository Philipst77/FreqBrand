"""
classify_wild.py — Run the trained ResNet-18 classifier on any model's spectra.

Used for:
  1. False-alarm test: Juggernaut-XL should score near 0 (not poisoned)
  2. Cross-logo test: HF-logo-poisoned model should score near 1 (poisoned)

Usage:
    python scripts/classify_wild.py \
        --spec_root  results/phase3_spectra/spectra \
        --test_name  juggernaut \
        --model_path results/phase3_detection/resnet18_classifier.pt \
        --out_dir    results/phase3_wild_classify
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def load_spectra_pool(spec_dir: Path) -> np.ndarray:
    paths = sorted(spec_dir.glob('*.npy'))
    if not paths:
        raise FileNotFoundError(f"No .npy spectra in {spec_dir}")
    sample = np.load(paths[0])
    pool = np.empty((len(paths), *sample.shape), dtype=np.float32)
    pool[0] = sample
    for i, p in enumerate(tqdm(paths[1:], desc=f'  Loading {spec_dir.name}', leave=False), 1):
        pool[i] = np.load(p)
    return pool


def bootstrap_aggregate(pool, ref_mean, sample_size, n_samples, rng):
    N, H, W = pool.shape
    out = np.empty((n_samples, 3, H, W), dtype=np.float32)
    for i in range(n_samples):
        idx   = rng.choice(N, size=sample_size, replace=True)
        batch = pool[idx]
        out[i, 0] = batch.mean(axis=0)
        out[i, 1] = batch.var(axis=0, ddof=1)
        out[i, 2] = batch.mean(axis=0) - ref_mean
    return out


def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def aggregate_to_tensor(aggregates, img_size=224):
    x = torch.from_numpy(aggregates)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode='bilinear', align_corners=False
    )
    for i in range(len(x)):
        for c in range(3):
            ch = x[i, c]
            x[i, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True)
    parser.add_argument('--test_name',   required=True,
                        help='Subdir name under spec_root (e.g. juggernaut, hf_logo_poisoned)')
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--n_bootstrap', type=int, default=300)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--img_size',    type=int, default=224)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.default_rng(99)

    print(f"\nLoading classifier from {args.model_path} ...")
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\nLoading spectra ...")
    pool_base = load_spectra_pool(spec_root / 'base')
    pool_test = load_spectra_pool(spec_root / args.test_name)
    ref_mean  = pool_base.mean(axis=0)
    print(f"  Base: {pool_base.shape}  |  {args.test_name}: {pool_test.shape}")

    print(f"\nBootstrapping {args.n_bootstrap} samples (N={args.sample_size}) ...")
    agg = bootstrap_aggregate(pool_test, ref_mean, args.sample_size, args.n_bootstrap, rng)

    tensor = aggregate_to_tensor(agg, args.img_size).to(device)
    scores, preds = [], []
    with torch.no_grad():
        for i in range(0, len(tensor), args.batch_size):
            logits = model(tensor[i:i + args.batch_size])
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    scores        = np.array(scores)
    preds         = np.array(preds)
    frac_poisoned = float(preds.mean())

    if frac_poisoned < 0.05:
        verdict = 'CLEAN'
    elif frac_poisoned > 0.95:
        verdict = 'POISONED'
    else:
        verdict = 'AMBIGUOUS'

    print(f"\n{'='*60}")
    print(f"RESULT: {args.test_name}")
    print(f"{'='*60}")
    print(f"  Fraction predicted POISONED: {frac_poisoned:.4f}  ({int(preds.sum())}/{len(preds)})")
    print(f"  Mean P(poisoned):            {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Verdict: {verdict}")

    report = {
        'test_name':     args.test_name,
        'n_bootstrap':   args.n_bootstrap,
        'sample_size':   args.sample_size,
        'frac_poisoned': round(frac_poisoned, 4),
        'mean_score':    round(float(scores.mean()), 4),
        'std_score':     round(float(scores.std()),  4),
        'min_score':     round(float(scores.min()),  4),
        'max_score':     round(float(scores.max()),  4),
        'verdict':       verdict,
    }
    rp = out_dir / f'{args.test_name}_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {rp}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=40, alpha=0.8, color='darkorange', density=True)
    ax.axvline(0.5, color='k', linestyle='--', linewidth=1, label='Decision boundary')
    ax.set_xlabel('P(poisoned)')
    ax.set_ylabel('Density')
    ax.set_title(f'{args.test_name} — classifier scores\n'
                 f'Fraction predicted poisoned: {frac_poisoned:.1%}  |  Verdict: {verdict}')
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fp = out_dir / f'{args.test_name}_scores.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"  Figure: {fp}")


if __name__ == '__main__':
    main()
