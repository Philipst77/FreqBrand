"""
ablation_population_size.py — How many images does FreqBrand need?

Uses the EXISTING trained classifier and existing 1K spectra pools.
For each N, generates 300 bootstrap samples and measures:
  - TPR: fraction of poisoned samples correctly flagged
  - FPR: fraction of clean samples incorrectly flagged

No retraining — this tests the deployed classifier's reliability as N varies.

Usage:
    python scripts/ablation_population_size.py \
        --spec_root  results/phase3_spectra/spectra \
        --model_path results/phase3_detection/resnet18_classifier.pt \
        --out_dir    results/ablation_population_size
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

N_VALUES = [25, 50, 100, 200, 500, 1000]
N_BOOTSTRAP = 300


def load_spectra_pool(spec_dir: Path) -> np.ndarray:
    paths = sorted(spec_dir.glob('*.npy'))
    if not paths:
        raise FileNotFoundError(f"No spectra in {spec_dir}")
    sample = np.load(paths[0])
    pool = np.empty((len(paths), *sample.shape), dtype=np.float32)
    pool[0] = sample
    for i, p in enumerate(tqdm(paths[1:], desc=f'  Loading {spec_dir.name}', leave=False), 1):
        pool[i] = np.load(p)
    return pool


def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def run_at_n(pool, ref_mean, n, n_bootstrap, model, device, rng, img_size=224):
    """Bootstrap n_bootstrap samples of size n, run classifier, return fraction predicted poisoned."""
    N, H, W = pool.shape
    aggregates = np.empty((n_bootstrap, 3, H, W), dtype=np.float32)
    for i in range(n_bootstrap):
        idx   = rng.choice(N, size=n, replace=True)
        batch = pool[idx]
        aggregates[i, 0] = batch.mean(axis=0)
        aggregates[i, 1] = batch.var(axis=0, ddof=1)
        aggregates[i, 2] = batch.mean(axis=0) - ref_mean

    x = torch.from_numpy(aggregates)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode='bilinear', align_corners=False
    )
    for i in range(len(x)):
        for c in range(3):
            ch = x[i, c]
            x[i, c] = (ch - ch.mean()) / (ch.std() + 1e-8)

    preds = []
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), 32):
            batch  = x[i:i+32].to(device)
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            scores.extend(probs.cpu().numpy())

    return float(np.mean(preds)), float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True)
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--out_dir',     required=True)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.default_rng(77)

    print("\nLoading classifier ...")
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("\nLoading spectra pools ...")
    pool_base     = load_spectra_pool(spec_root / 'base')
    pool_clean    = load_spectra_pool(spec_root / 'clean')
    pool_poisoned = load_spectra_pool(spec_root / 'poisoned')
    ref_mean      = pool_base.mean(axis=0)
    print(f"  Base={pool_base.shape}  Clean={pool_clean.shape}  Poisoned={pool_poisoned.shape}")

    results = {}
    print(f"\nRunning ablation over N = {N_VALUES} ...")
    for n in N_VALUES:
        print(f"\n  N = {n} ...")
        tpr, tpr_score = run_at_n(pool_poisoned, ref_mean, n, N_BOOTSTRAP, model, device, rng)
        fpr, fpr_score = run_at_n(pool_clean,    ref_mean, n, N_BOOTSTRAP, model, device, rng)
        results[n] = {
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4),
            'mean_score_poisoned': round(tpr_score, 4),
            'mean_score_clean':    round(fpr_score, 4),
        }
        print(f"    TPR={tpr:.4f}  FPR={fpr:.4f}  "
              f"(mean_poisoned={tpr_score:.4f}  mean_clean={fpr_score:.4f})")

    print(f"\n{'='*60}")
    print("POPULATION SIZE ABLATION RESULTS")
    print(f"{'='*60}")
    print(f"{'N':>6}  {'TPR':>8}  {'FPR':>8}")
    for n, r in results.items():
        print(f"{n:>6}  {r['tpr']:>8.4f}  {r['fpr']:>8.4f}")

    report_path = out_dir / 'population_size_ablation.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport: {report_path}")

    # Plot
    ns   = list(results.keys())
    tprs = [results[n]['tpr'] for n in ns]
    fprs = [results[n]['fpr'] for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, tprs, 'o-', color='crimson',    label='TPR (poisoned detected)', linewidth=2)
    ax.plot(ns, fprs, 's--', color='steelblue', label='FPR (clean mis-flagged)', linewidth=2)
    ax.axhline(1.0, color='crimson',    alpha=0.2, linestyle=':')
    ax.axhline(0.05, color='steelblue', alpha=0.2, linestyle=':', label='FPR=5% threshold')
    ax.set_xscale('log')
    ax.set_xlabel('Population size N (images)')
    ax.set_ylabel('Rate')
    ax.set_title('FreqBrand detection rate vs. population size')
    ax.legend()
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    plt.tight_layout()
    fig_path = out_dir / 'population_size_ablation.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure: {fig_path}")


if __name__ == '__main__':
    main()
