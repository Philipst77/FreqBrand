"""
retrain_classifier_diverse.py — Retrain ResNet-18 with diverse clean negatives.

Problem: original classifier trained with only one clean negative (clean LoRA,
same training procedure). Juggernaut-XL (full fine-tune, massive dataset) has
a different spectral signature and gets false-alarmed at 99.7%.

Fix: add Juggernaut spectra to the clean pool so the classifier learns to
ignore general fine-tuning artifacts and only flag the logo fingerprint.

Training data:
  Positive (label=1): poisoned LoRA (Avengers logo)
  Negative (label=0): clean LoRA  +  Juggernaut-XL  (both legitimate, no logo)

Usage:
    python scripts/retrain_classifier_diverse.py \
        --spec_root  results/phase3_spectra/spectra \
        --out_dir    results/phase3_detection_diverse
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


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


def bootstrap_aggregate(pool, ref_mean, sample_size, n_samples, rng):
    N, H, W = pool.shape
    out = np.empty((n_samples, 3, H, W), dtype=np.float32)
    for i in range(n_samples):
        idx      = rng.choice(N, size=sample_size, replace=True)
        batch    = pool[idx]
        s_mean   = batch.mean(axis=0)
        out[i, 0] = s_mean
        out[i, 1] = batch.var(axis=0, ddof=1)
        out[i, 2] = s_mean - ref_mean
    return out


class SpectralDataset(Dataset):
    def __init__(self, aggregates, labels, img_size=224):
        self.data   = torch.from_numpy(aggregates)
        self.labels = torch.from_numpy(labels).long()
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        for c in range(3):
            ch = x[c]
            x[c] = (ch - ch.mean()) / (ch.std() + 1e-8)
        return x, self.labels[idx]


def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_resnet(train_loader, val_loader, device, epochs=30, lr=1e-3):
    model     = build_resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_auc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
                vl.extend(y.numpy())
        auc = roc_auc_score(vl, vp) if len(set(vl)) > 1 else 0.5
        print(f"  Epoch {epoch+1:02d}/{epochs} — val AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate(model, loader, device):
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out[:, 1].cpu().numpy())
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, zero_division=0)
    return {'auroc': round(auc, 4), 'accuracy': round(acc, 4), 'f1': round(f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True,
                        help='Contains base/, clean/, poisoned/, juggernaut/ subdirs')
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--n_bootstrap', type=int, default=500)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.default_rng(42)

    print("\nLoading spectra pools ...")
    pool_base      = load_spectra_pool(spec_root / 'base')
    pool_clean     = load_spectra_pool(spec_root / 'clean')
    pool_poisoned  = load_spectra_pool(spec_root / 'poisoned')
    pool_jugg      = load_spectra_pool(spec_root / 'juggernaut')
    ref_mean       = pool_base.mean(axis=0)
    print(f"  Base={pool_base.shape}  Clean={pool_clean.shape}  "
          f"Poisoned={pool_poisoned.shape}  Juggernaut={pool_jugg.shape}")

    print(f"\nBootstrapping {args.n_bootstrap} samples (N={args.sample_size}) per class ...")
    agg_poisoned = bootstrap_aggregate(pool_poisoned, ref_mean,
                                       args.sample_size, args.n_bootstrap, rng)
    # Diverse clean pool: equal split between clean LoRA and Juggernaut
    half = args.n_bootstrap // 2
    agg_clean = bootstrap_aggregate(pool_clean, ref_mean, args.sample_size, half, rng)
    agg_jugg  = bootstrap_aggregate(pool_jugg,  ref_mean, args.sample_size, half, rng)
    agg_neg   = np.concatenate([agg_clean, agg_jugg], axis=0)

    print(f"  Positives (poisoned):  {len(agg_poisoned)}")
    print(f"  Negatives (clean LoRA + Juggernaut):  {len(agg_neg)} "
          f"({half} + {half})")

    X = np.concatenate([agg_poisoned, agg_neg], axis=0)
    y = np.array([1] * len(agg_poisoned) + [0] * len(agg_neg))

    dataset = SpectralDataset(X, y)
    n       = len(dataset)
    n_test  = int(n * 0.15)
    n_val   = int(n * 0.15)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("\nTraining ResNet-18 (diverse clean pool) ...")
    model = train_resnet(train_loader, val_loader, device, epochs=args.epochs)

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest (poisoned vs diverse clean): "
          f"AUROC={test_metrics['auroc']:.4f}  "
          f"Acc={test_metrics['accuracy']:.4f}  F1={test_metrics['f1']:.4f}")

    # Re-evaluate on individual clean pools to check FPR breakdown
    clean_ds  = SpectralDataset(agg_clean, np.zeros(len(agg_clean), dtype=int))
    jugg_ds   = SpectralDataset(agg_jugg,  np.zeros(len(agg_jugg),  dtype=int))
    pois_ds   = SpectralDataset(agg_poisoned, np.ones(len(agg_poisoned), dtype=int))
    clean_m   = evaluate(model, DataLoader(clean_ds,  batch_size=32, num_workers=2), device)
    jugg_m    = evaluate(model, DataLoader(jugg_ds,   batch_size=32, num_workers=2), device)
    pois_m    = evaluate(model, DataLoader(pois_ds,   batch_size=32, num_workers=2), device)

    print(f"\nPer-pool results:")
    print(f"  Clean LoRA   FPR: {1 - clean_m['accuracy']:.4f}  (accuracy={clean_m['accuracy']:.4f})")
    print(f"  Juggernaut   FPR: {1 - jugg_m['accuracy']:.4f}   (accuracy={jugg_m['accuracy']:.4f})")
    print(f"  Poisoned     TPR: {pois_m['accuracy']:.4f}")

    model_path = out_dir / 'resnet18_diverse_classifier.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    metrics = {
        'training_negatives': ['clean_lora', 'juggernaut'],
        'n_bootstrap': args.n_bootstrap,
        'sample_size': args.sample_size,
        'test_overall':     test_metrics,
        'fpr_clean_lora':   round(1 - clean_m['accuracy'], 4),
        'fpr_juggernaut':   round(1 - jugg_m['accuracy'],  4),
        'tpr_poisoned':     round(pois_m['accuracy'],       4),
    }
    rp = out_dir / 'diverse_classifier_metrics.json'
    with open(rp, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Report: {rp}")

    # Score distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, pool, name, color in zip(
        axes,
        [agg_clean, agg_jugg, agg_poisoned],
        ['Clean LoRA', 'Juggernaut', 'Poisoned'],
        ['steelblue', 'seagreen', 'crimson']
    ):
        ds  = SpectralDataset(pool, np.zeros(len(pool), dtype=int))
        dl  = DataLoader(ds, batch_size=32, num_workers=2)
        sc  = []
        model.eval()
        with torch.no_grad():
            for x, _ in dl:
                sc.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
        sc = np.array(sc)
        ax.hist(sc, bins=30, color=color, alpha=0.8, density=True)
        ax.axvline(0.5, color='k', linestyle='--', linewidth=1)
        ax.set_title(f'{name}\nMean={sc.mean():.3f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('P(poisoned)')
    plt.suptitle('Diverse-trained classifier score distributions')
    plt.tight_layout()
    fp = out_dir / 'diverse_classifier_scores.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Figure: {fp}")


if __name__ == '__main__':
    main()
