"""
ablation_aggregation.py — Compare aggregation methods: mean vs median vs trimmed mean.

Trains a fresh ResNet-18 for each aggregation variant and compares AUROC.
Uses existing 1K DCT spectra — no new image generation needed.

Usage:
    python scripts/ablation_aggregation.py \
        --spec_root results/phase3_spectra/spectra \
        --out_dir   results/ablation_aggregation
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

N_BOOTSTRAP = 400
SAMPLE_SIZE = 100
TRIM_FRAC   = 0.1   # trim top/bottom 10% for trimmed mean


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


def agg_mean(batch):
    return batch.mean(axis=0)


def agg_median(batch):
    return np.median(batch, axis=0)


def agg_trimmed_mean(batch, trim=TRIM_FRAC):
    n = batch.shape[0]
    k = max(1, int(n * trim))
    sorted_b = np.sort(batch, axis=0)
    return sorted_b[k:-k].mean(axis=0)


AGG_METHODS = {
    'mean':         agg_mean,
    'median':       agg_median,
    'trimmed_mean': agg_trimmed_mean,
}


def bootstrap_aggregate(pool, ref_mean, sample_size, n_samples, rng, agg_fn):
    N, H, W = pool.shape
    out = np.empty((n_samples, 3, H, W), dtype=np.float32)
    for i in range(n_samples):
        idx   = rng.choice(N, size=sample_size, replace=True)
        batch = pool[idx]
        s_agg     = agg_fn(batch)
        s_var     = batch.var(axis=0, ddof=1)
        delta     = s_agg - ref_mean
        out[i, 0] = s_agg
        out[i, 1] = s_var
        out[i, 2] = delta
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


def train_and_eval(agg_poisoned, agg_clean, device, epochs=20):
    labels = np.array([1] * len(agg_poisoned) + [0] * len(agg_clean))
    data   = np.concatenate([agg_poisoned, agg_clean], axis=0)

    dataset = SpectralDataset(data, labels)
    n       = len(dataset)
    n_test  = int(n * 0.15)
    n_val   = int(n * 0.15)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

    model     = build_resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(y.numpy())
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_labels.extend(y.numpy())
    test_auc = roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5
    return round(test_auc, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root', required=True)
    parser.add_argument('--out_dir',   required=True)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading spectra pools ...")
    pool_base     = load_spectra_pool(spec_root / 'base')
    pool_clean    = load_spectra_pool(spec_root / 'clean')
    pool_poisoned = load_spectra_pool(spec_root / 'poisoned')
    ref_mean      = pool_base.mean(axis=0)

    results = {}
    for method_name, agg_fn in AGG_METHODS.items():
        print(f"\n[{method_name}] bootstrapping and training ...")
        rng = np.random.default_rng(42)
        agg_p = bootstrap_aggregate(pool_poisoned, ref_mean, SAMPLE_SIZE, N_BOOTSTRAP, rng, agg_fn)
        agg_c = bootstrap_aggregate(pool_clean,    ref_mean, SAMPLE_SIZE, N_BOOTSTRAP, rng, agg_fn)
        auc   = train_and_eval(agg_p, agg_c, device)
        results[method_name] = {'auroc': auc}
        print(f"  AUROC = {auc:.4f}")

    print(f"\n{'='*50}")
    print("AGGREGATION ABLATION RESULTS")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:<20}: AUROC = {v['auroc']:.4f}")

    report_path = out_dir / 'aggregation_ablation.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport: {report_path}")

    # Bar chart
    methods = list(results.keys())
    aurocs  = [results[m]['auroc'] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, aurocs, color=['steelblue', 'seagreen', 'darkorange'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title('FreqBrand: Aggregation method ablation')
    for bar, auc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig_path = out_dir / 'aggregation_ablation.png'
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure: {fig_path}")


if __name__ == '__main__':
    main()
