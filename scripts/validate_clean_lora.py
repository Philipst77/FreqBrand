"""
validate_clean_lora.py — Priority 1 validation

Loads the trained ResNet-18 classifier and explicitly reports the false
positive rate (FPR) on clean LoRA bootstrap aggregates.

The classifier was trained with:
  poisoned LoRA  →  label 1
  clean LoRA     →  label 0

This script runs inference on bootstrap samples drawn from:
  - clean LoRA pool   → should all predict 0 (not poisoned), FPR should be ≈ 0
  - poisoned LoRA pool → should all predict 1 (poisoned), sanity check TPR ≈ 1

Usage:
    python scripts/validate_clean_lora.py \
        --spec_root  results/phase3_spectra/spectra \
        --model_path results/phase3_detection/resnet18_classifier.pt \
        --out_dir    results/phase3_validation
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


# ---------------------------------------------------------------------------
# Helpers — identical to train_classifier.py so features match exactly
# ---------------------------------------------------------------------------

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


def bootstrap_aggregate(pool: np.ndarray, ref_mean: np.ndarray,
                        sample_size: int, n_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    N, H, W = pool.shape
    out = np.empty((n_samples, 3, H, W), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(N, size=sample_size, replace=True)
        batch = pool[idx]
        s_mean = batch.mean(axis=0)
        s_var  = batch.var(axis=0, ddof=1)
        delta  = s_mean - ref_mean
        out[i, 0] = s_mean
        out[i, 1] = s_var
        out[i, 2] = delta
    return out


def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def aggregate_to_tensor(aggregates: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Convert (N, 3, H, W) numpy array to normalized (N, 3, img_size, img_size) tensor."""
    x = torch.from_numpy(aggregates)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode='bilinear', align_corners=False
    )
    # Normalize each channel independently (same as SpectralDataset)
    for i in range(len(x)):
        for c in range(3):
            ch = x[i, c]
            x[i, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True)
    parser.add_argument('--model_path',  required=True,
                        help='Path to resnet18_classifier.pt')
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--n_bootstrap', type=int, default=300)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--img_size',    type=int, default=224)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    spec_root  = Path(args.spec_root)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.default_rng(0)   # different seed from training

    # -----------------------------------------------------------------------
    # Load classifier
    # -----------------------------------------------------------------------
    print(f"\nLoading ResNet-18 from {args.model_path} ...")
    model = build_resnet18().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("  Classifier loaded OK.")

    # -----------------------------------------------------------------------
    # Load spectra pools
    # -----------------------------------------------------------------------
    print("\nLoading spectra pools ...")
    pool_base     = load_spectra_pool(spec_root / 'base')
    pool_clean    = load_spectra_pool(spec_root / 'clean')
    pool_poisoned = load_spectra_pool(spec_root / 'poisoned')

    ref_mean = pool_base.mean(axis=0)
    print(f"  Base: {pool_base.shape}  |  Clean: {pool_clean.shape}  "
          f"|  Poisoned: {pool_poisoned.shape}")

    # -----------------------------------------------------------------------
    # Bootstrap aggregates
    # -----------------------------------------------------------------------
    print(f"\nBootstrapping {args.n_bootstrap} samples (N={args.sample_size}) "
          f"from clean LoRA and poisoned LoRA ...")
    agg_clean    = bootstrap_aggregate(pool_clean,    ref_mean, args.sample_size,
                                       args.n_bootstrap, rng)
    agg_poisoned = bootstrap_aggregate(pool_poisoned, ref_mean, args.sample_size,
                                       args.n_bootstrap, rng)

    # -----------------------------------------------------------------------
    # Run classifier
    # -----------------------------------------------------------------------
    def run_inference(aggregates: np.ndarray, label_name: str):
        tensor = aggregate_to_tensor(aggregates, args.img_size).to(device)
        scores, preds = [], []
        with torch.no_grad():
            for i in range(0, len(tensor), args.batch_size):
                batch = tensor[i:i + args.batch_size]
                logits = model(batch)
                probs  = torch.softmax(logits, dim=1)[:, 1]   # prob of "poisoned"
                scores.extend(probs.cpu().numpy().tolist())
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        scores = np.array(scores)
        preds  = np.array(preds)
        frac_poisoned = preds.mean()
        print(f"\n  [{label_name}]")
        print(f"    Fraction predicted POISONED: {frac_poisoned:.4f}  "
              f"({preds.sum()}/{len(preds)})")
        print(f"    Mean score (P(poisoned)):    {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"    Min / Max score:             {scores.min():.4f} / {scores.max():.4f}")
        return scores, preds, frac_poisoned

    print("\nRunning inference ...")
    clean_scores,    clean_preds,    clean_fpr    = run_inference(agg_clean,    "clean LoRA    (should predict 0 = NOT poisoned)")
    poisoned_scores, poisoned_preds, poisoned_tpr = run_inference(agg_poisoned, "poisoned LoRA (should predict 1 = POISONED)")

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEAN LoRA FPR VALIDATION RESULTS")
    print("=" * 60)
    print(f"  False Positive Rate (clean LoRA predicted as poisoned): {clean_fpr:.4f}")
    print(f"  True  Positive Rate (poisoned LoRA correctly flagged):  {poisoned_tpr:.4f}")

    if clean_fpr < 0.05:
        print("\n  ✓ PASSED: FPR < 5% — classifier learned logo fingerprint, not finetuning artifacts.")
    elif clean_fpr < 0.15:
        print("\n  ~ MARGINAL: FPR between 5–15% — some confusion with finetuning artifacts.")
    else:
        print("\n  ✗ FAILED: FPR > 15% — classifier may be detecting finetuning, not the logo.")

    # -----------------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------------
    report = {
        'clean_lora': {
            'false_positive_rate': round(float(clean_fpr), 4),
            'mean_score': round(float(clean_scores.mean()), 4),
            'std_score':  round(float(clean_scores.std()),  4),
            'n_bootstrap': args.n_bootstrap,
            'sample_size': args.sample_size,
        },
        'poisoned_lora': {
            'true_positive_rate': round(float(poisoned_tpr), 4),
            'mean_score': round(float(poisoned_scores.mean()), 4),
            'std_score':  round(float(poisoned_scores.std()),  4),
        },
        'verdict': 'PASS' if clean_fpr < 0.05 else ('MARGINAL' if clean_fpr < 0.15 else 'FAIL'),
    }
    report_path = out_dir / 'clean_lora_fpr_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # -----------------------------------------------------------------------
    # Plot score distributions
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clean_scores,    bins=40, alpha=0.7, color='steelblue',
            label=f'Clean LoRA (FPR={clean_fpr:.3f})',    density=True)
    ax.hist(poisoned_scores, bins=40, alpha=0.7, color='crimson',
            label=f'Poisoned LoRA (TPR={poisoned_tpr:.3f})', density=True)
    ax.axvline(0.5, color='k', linestyle='--', linewidth=1, label='Decision boundary (0.5)')
    ax.set_xlabel('P(poisoned) — classifier score')
    ax.set_ylabel('Density')
    ax.set_title('Clean LoRA vs Poisoned LoRA — Classifier Score Distribution\n'
                 '(clean should cluster near 0, poisoned near 1)')
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig_path = out_dir / 'clean_lora_fpr.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")


if __name__ == '__main__':
    main()
