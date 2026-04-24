"""
train_classifier.py  — Phase 3 Step 2

FreqBrand detection classifier.

Approach: bootstrap sampling over per-image spectra to generate multiple
population-level training examples from a single model's image pool.

For each bootstrap sample:
  - Draw N spectra at random (with replacement) from the model's pool
  - Compute S_mean, S_var (population-level aggregates)
  - Compute delta_S = S_mean_target - S_mean_base_ref
  - Stack → 3-channel input of shape (3, H, W)

Labels:
  - poisoned model → 1
  - clean LoRA     → 0  (also used as held-out test of generalization)
  - base SDXL      → 0

Two classifiers:
  1. Linear baseline: radially-averaged spectrum → 512-dim feature → LogisticRegression
  2. ResNet-18: 3-channel aggregate image → binary classification

Usage:
    python scripts/train_classifier.py \
        --spec_root results/phase3_spectra/spectra \
        --out_dir   results/phase3_detection \
        --n_bootstrap 500 \
        --sample_size 100
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
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torchvision.models as models

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Bootstrap aggregate dataset
# ---------------------------------------------------------------------------

def load_spectra_pool(spec_dir: Path) -> np.ndarray:
    """Load all .npy spectra from a directory into (N, H, W) array."""
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
    """
    Generate n_samples bootstrap aggregates from pool.
    Each aggregate: (3, H, W) = [S_mean, S_var, delta_S]
    Returns: (n_samples, 3, H, W) float32
    """
    N, H, W = pool.shape
    out = np.empty((n_samples, 3, H, W), dtype=np.float32)
    for i in range(n_samples):
        idx = rng.choice(N, size=sample_size, replace=True)
        batch = pool[idx]           # (sample_size, H, W)
        s_mean = batch.mean(axis=0)
        s_var  = batch.var(axis=0, ddof=1)
        delta  = s_mean - ref_mean
        out[i, 0] = s_mean
        out[i, 1] = s_var
        out[i, 2] = delta
    return out


class SpectralDataset(Dataset):
    def __init__(self, aggregates: np.ndarray, labels: np.ndarray, img_size: int = 224):
        self.data   = torch.from_numpy(aggregates)   # (N, 3, H, W)
        self.labels = torch.from_numpy(labels).long()
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]          # (3, H, W)
        # Resize to img_size via interpolation
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        # Normalize each channel independently
        for c in range(3):
            ch = x[c]
            x[c] = (ch - ch.mean()) / (ch.std() + 1e-8)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# Linear baseline — radial spectral statistics
# ---------------------------------------------------------------------------

def radial_features(aggregates: np.ndarray, n_bins: int = 512) -> np.ndarray:
    """
    For each aggregate (3, H, W), compute radially-averaged spectrum per channel.
    Returns (N, 3 * n_bins) feature matrix.
    """
    N, C, H, W = aggregates.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.float32)
    r_max = r.max()

    features = np.empty((N, C * n_bins), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            channel = aggregates[i, c]
            bins = np.linspace(0, r_max, n_bins + 1)
            profile = np.zeros(n_bins, dtype=np.float32)
            for b in range(n_bins):
                mask = (r >= bins[b]) & (r < bins[b + 1])
                if mask.any():
                    profile[b] = channel[mask].mean()
            features[i, c * n_bins:(c + 1) * n_bins] = profile
    return features


# ---------------------------------------------------------------------------
# ResNet-18 classifier
# ---------------------------------------------------------------------------

def build_resnet18():
    model = models.resnet18(weights=None)
    # Adapt first conv: ImageNet has 3 channels, we have 3 (S_mean, S_var, delta_S)
    # so the architecture is unchanged — just don't use ImageNet weights
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_resnet(train_loader, val_loader, device, epochs=20, lr=1e-3):
    model = build_resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        print(f"  Epoch {epoch+1:02d}/{epochs} — loss: {train_loss/len(train_loader):.4f} "
              f"| val AUROC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate(model_or_clf, loader_or_X, y_true, device=None, is_sklearn=False):
    if is_sklearn:
        probs = model_or_clf.predict_proba(loader_or_X)[:, 1]
        preds = model_or_clf.predict(loader_or_X)
    else:
        model_or_clf.eval()
        probs, preds = [], []
        with torch.no_grad():
            for x, _ in loader_or_X:
                x = x.to(device)
                out = torch.softmax(model_or_clf(x), dim=1)
                probs.extend(out[:, 1].cpu().numpy())
                preds.extend(out.argmax(dim=1).cpu().numpy())

    auc  = roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float('nan')
    acc  = accuracy_score(y_true, preds if is_sklearn else preds)
    f1   = f1_score(y_true, preds if is_sklearn else preds, zero_division=0)
    return {'auroc': round(auc, 4), 'accuracy': round(acc, 4), 'f1': round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True,
                        help='Root dir containing base/, clean/, poisoned/ spectra subdirs')
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--n_bootstrap', type=int, default=500,
                        help='Bootstrap samples per model class (default: 500)')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Images per bootstrap sample (default: 100)')
    parser.add_argument('--epochs',      type=int, default=30)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--img_size',    type=int, default=224)
    parser.add_argument('--val_frac',    type=float, default=0.15)
    parser.add_argument('--test_frac',   type=float, default=0.15)
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    # Load spectra pools
    # -----------------------------------------------------------------------
    print("\nLoading spectra pools ...")
    pool_base     = load_spectra_pool(spec_root / 'base')
    pool_clean    = load_spectra_pool(spec_root / 'clean')
    pool_poisoned = load_spectra_pool(spec_root / 'poisoned')

    # Reference mean = full base pool mean (fixed, not bootstrapped)
    ref_mean = pool_base.mean(axis=0)
    print(f"  Base pool:     {pool_base.shape}")
    print(f"  Clean pool:    {pool_clean.shape}")
    print(f"  Poisoned pool: {pool_poisoned.shape}")

    # -----------------------------------------------------------------------
    # Bootstrap aggregates
    # -----------------------------------------------------------------------
    print(f"\nGenerating {args.n_bootstrap} bootstrap samples "
          f"(N={args.sample_size}) per class ...")

    agg_poisoned = bootstrap_aggregate(pool_poisoned, ref_mean,
                                       args.sample_size, args.n_bootstrap, rng)
    agg_base     = bootstrap_aggregate(pool_base, ref_mean,
                                       args.sample_size, args.n_bootstrap, rng)
    agg_clean    = bootstrap_aggregate(pool_clean, ref_mean,
                                       args.sample_size, args.n_bootstrap, rng)

    # Training data: poisoned (1) vs clean LoRA (0)
    # Both are LoRA-finetuned — only systematic difference is the logo.
    # Base SDXL is used only as the delta_S reference, not as a training class.
    X_train_agg = np.concatenate([agg_poisoned, agg_clean], axis=0)
    y_train_arr = np.array([1] * args.n_bootstrap + [0] * args.n_bootstrap)

    # Held-out generalization test: base SDXL (should be predicted 0 — not poisoned)
    X_clean_agg = agg_base
    y_clean_arr = np.zeros(args.n_bootstrap, dtype=int)

    print(f"  Training set: {X_train_agg.shape} | labels: {y_train_arr.sum()} poisoned, "
          f"{(y_train_arr==0).sum()} clean LoRA")

    # -----------------------------------------------------------------------
    # 1. Linear baseline
    # -----------------------------------------------------------------------
    print("\n[1/2] Linear baseline (radial features → LogisticRegression) ...")
    print("  Computing radial features (this may take a few minutes) ...")
    X_rad_train = radial_features(X_train_agg)
    X_rad_clean = radial_features(X_clean_agg)

    scaler = StandardScaler()
    X_rad_train_s = scaler.fit_transform(X_rad_train)
    X_rad_clean_s = scaler.transform(X_rad_clean)

    # Train/test split for linear baseline
    n_total  = len(y_train_arr)
    n_test   = int(n_total * args.test_frac)
    n_val    = int(n_total * args.val_frac)
    idx_all  = np.random.permutation(n_total)
    idx_test = idx_all[:n_test]
    idx_rest = idx_all[n_test:]

    lr_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_clf.fit(X_rad_train_s[idx_rest], y_train_arr[idx_rest])

    lin_test  = evaluate(lr_clf, X_rad_train_s[idx_test], y_train_arr[idx_test],
                         is_sklearn=True)
    lin_clean = evaluate(lr_clf, X_rad_clean_s, y_clean_arr, is_sklearn=True)

    print(f"  Test  (poisoned vs clean LoRA): AUROC={lin_test['auroc']:.4f}  "
          f"Acc={lin_test['accuracy']:.4f}  F1={lin_test['f1']:.4f}")
    print(f"  Base SDXL generalization: AUROC={lin_clean['auroc']:.4f}  "
          f"Acc={lin_clean['accuracy']:.4f}  F1={lin_clean['f1']:.4f}")

    # -----------------------------------------------------------------------
    # 2. ResNet-18
    # -----------------------------------------------------------------------
    print("\n[2/2] ResNet-18 classifier ...")
    dataset = SpectralDataset(X_train_agg, y_train_arr, img_size=args.img_size)
    n_total  = len(dataset)
    n_test   = int(n_total * args.test_frac)
    n_val    = int(n_total * args.val_frac)
    n_train  = n_total - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    clean_ds     = SpectralDataset(X_clean_agg, y_clean_arr, img_size=args.img_size)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = train_resnet(train_loader, val_loader, device,
                         epochs=args.epochs, lr=args.lr)

    resnet_test  = evaluate(model, test_loader,  list(y_train_arr[
        list(test_ds.indices)]), device)
    resnet_clean = evaluate(model, clean_loader, list(y_clean_arr), device)

    print(f"\n  Test  (poisoned vs clean LoRA): AUROC={resnet_test['auroc']:.4f}  "
          f"Acc={resnet_test['accuracy']:.4f}  F1={resnet_test['f1']:.4f}")
    print(f"  Base SDXL generalization: AUROC={resnet_clean['auroc']:.4f}  "
          f"Acc={resnet_clean['accuracy']:.4f}  F1={resnet_clean['f1']:.4f}")

    # Save model
    model_path = out_dir / 'resnet18_classifier.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    # -----------------------------------------------------------------------
    # Save all metrics
    # -----------------------------------------------------------------------
    metrics = {
        'config': {
            'n_bootstrap': args.n_bootstrap,
            'sample_size': args.sample_size,
            'epochs': args.epochs,
            'img_size': args.img_size,
        },
        'linear_baseline': {
            'test_poisoned_vs_clean_lora': lin_test,
            'generalization_base_sdxl': lin_clean,
        },
        'resnet18': {
            'test_poisoned_vs_clean_lora': resnet_test,
            'generalization_base_sdxl': resnet_clean,
        },
    }

    metrics_path = out_dir / 'classifier_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("CLASSIFIER TRAINING COMPLETE")
    print(f"  Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    print("=" * 60)


if __name__ == '__main__':
    main()
