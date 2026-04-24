"""
validate_classifier.py — Comprehensive FreqBrand detector validation

Every test exists to answer one question:
  "Is the AUROC=1.0 real, or did something cheat?"

Tests run:
  1. N Ablation          — AUROC vs population size (is aggregation doing real work?)
  2. Permutation test    — shuffle labels, retrain → must collapse to ~0.5 (gold standard)
  3. Channel ablation    — which channel(s) carry the signal?
  4. K-fold (image pool) — proper independence: test images never seen during training
  5. Per-image test      — can individual spectra be classified? (if yes, aggregation isn't needed)
  6. Frequency masking   — where in DCT space does the signal live?
  7. Bootstrap overlap   — how correlated are our train/test samples?
  8. Seed stability      — does AUROC hold under different random seeds?
  9. DC sanity check     — are poisoned/clean images just brighter/darker?

Usage:
    python scripts/validate_classifier.py \
        --spec_root results/phase3_spectra/spectra \
        --out_dir   results/phase3_validation
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def downsample_pool(pool: np.ndarray, size: int) -> np.ndarray:
    """Downsample (N, H, W) pool to (N, size, size) using area averaging."""
    if pool.shape[1] == size:
        return pool
    t = torch.from_numpy(pool[:, np.newaxis, :, :])   # (N,1,H,W)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode='area')
    return t[:, 0, :, :].numpy()


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
                        rng: np.random.Generator,
                        channel_mask: list = None) -> np.ndarray:
    """
    Generate n_samples bootstrap aggregates from pool.
    Each: (3, H, W) = [S_mean, S_var, delta_S]
    channel_mask: list of channel indices to keep (default: all [0,1,2])
    """
    N, H, W = pool.shape
    channels = channel_mask if channel_mask is not None else [0, 1, 2]
    out = np.empty((n_samples, len(channels), H, W), dtype=np.float32)
    all_channels = []
    for i in range(n_samples):
        idx = rng.choice(N, size=sample_size, replace=True)
        batch = pool[idx]
        s_mean = batch.mean(axis=0)
        s_var  = batch.var(axis=0, ddof=1)
        delta  = s_mean - ref_mean
        full   = [s_mean, s_var, delta]
        for j, c in enumerate(channels):
            out[i, j] = full[c]
    return out


# ---------------------------------------------------------------------------
# Fast radial features (O(H*W) not O(H*W*bins))
# ---------------------------------------------------------------------------

def precompute_bins(H: int, W: int, n_bins: int = 256):
    """Precompute radial bin assignment for each pixel. Returns (H*W,) int32."""
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).flatten().astype(np.float32)
    edges = np.linspace(0, r.max() + 1e-6, n_bins + 1)
    bin_idx = np.digitize(r, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1).astype(np.int32)
    return bin_idx


def radial_features_fast(aggregates: np.ndarray, bin_idx: np.ndarray,
                         n_bins: int = 256) -> np.ndarray:
    """
    Compute radial profile features for (N, C, H, W) aggregates.
    Returns (N, C * n_bins) feature matrix.
    """
    N, C, H, W = aggregates.shape
    features = np.empty((N, C * n_bins), dtype=np.float32)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
    counts = np.where(counts > 0, counts, 1.0)  # avoid divide-by-zero
    for i in range(N):
        for c in range(C):
            channel = aggregates[i, c].flatten()
            sums = np.bincount(bin_idx, weights=channel, minlength=n_bins)
            features[i, c * n_bins:(c + 1) * n_bins] = (sums / counts).astype(np.float32)
    return features


def train_linear(X_train, y_train):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(Xs, y_train)
    return clf, scaler


def auroc_linear(clf, scaler, X_test, y_test):
    Xs = scaler.transform(X_test)
    if len(set(y_test)) < 2:
        return float('nan')
    probs = clf.predict_proba(Xs)[:, 1]
    return roc_auc_score(y_test, probs)


# ---------------------------------------------------------------------------
# ResNet-18 (for K-fold only)
# ---------------------------------------------------------------------------

class SpectralDataset(Dataset):
    def __init__(self, aggregates: np.ndarray, labels: np.ndarray, img_size: int = 224):
        self.data   = torch.from_numpy(aggregates)
        self.labels = torch.from_numpy(labels).long()
        self.size   = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].float()
        x = nn.functional.interpolate(x.unsqueeze(0), (self.size, self.size),
                                       mode='bilinear', align_corners=False).squeeze(0)
        for c in range(x.shape[0]):
            x[c] = (x[c] - x[c].mean()) / (x[c].std() + 1e-8)
        return x, self.labels[idx]


def train_resnet(X: np.ndarray, y: np.ndarray, device, epochs=20, img_size=224):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ds = SpectralDataset(X, y, img_size)
    n_val = max(1, int(0.15 * len(ds)))
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val],
                                  generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=32, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    best_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy()
                probs.extend(p); labels.extend(yb.numpy())
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def eval_resnet(model, X: np.ndarray, y: np.ndarray, device, img_size=224):
    ds = SpectralDataset(X, y, img_size)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy()
            probs.extend(p); labels.extend(yb.numpy())
    return roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_root',   required=True)
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--n_bootstrap', type=int, default=300,
                        help='Bootstrap samples per class for most tests')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Default images per bootstrap sample')
    parser.add_argument('--n_perms',     type=int, default=100,
                        help='Number of permutations for permutation test')
    parser.add_argument('--n_bins',      type=int, default=256,
                        help='Radial bins for feature extraction')
    parser.add_argument('--kfolds',      type=int, default=5)
    parser.add_argument('--downsample',  type=int, default=256,
                        help='Downsample spectra to NxN before aggregation (saves memory, 0=no downsample)')
    args = parser.parse_args()

    spec_root = Path(args.spec_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(42)

    results = {}
    t0_total = time.time()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("FreqBrand — Comprehensive Validation")
    print("=" * 65)
    print("\nLoading spectra pools ...")
    pool_base     = load_spectra_pool(spec_root / 'base')
    pool_clean    = load_spectra_pool(spec_root / 'clean')
    pool_poisoned = load_spectra_pool(spec_root / 'poisoned')
    ref_mean = pool_base.mean(axis=0)
    H, W = pool_base.shape[1], pool_base.shape[2]

    print(f"  Base:     {pool_base.shape}")
    print(f"  Clean:    {pool_clean.shape}")
    print(f"  Poisoned: {pool_poisoned.shape}")

    ref_mean = pool_base.mean(axis=0)   # provisional — recomputed after downsample if needed

    if args.downsample > 0 and pool_base.shape[1] != args.downsample:
        print(f"\nDownsampling spectra {pool_base.shape[1]}→{args.downsample} (memory reduction) ...")
        pool_base     = downsample_pool(pool_base,     args.downsample)
        pool_clean    = downsample_pool(pool_clean,    args.downsample)
        pool_poisoned = downsample_pool(pool_poisoned, args.downsample)
        gc.collect()
        print(f"  New shape: {pool_base.shape}")

    H, W = pool_base.shape[1], pool_base.shape[2]
    print(f"  Working spectrum shape: {H}x{W}")

    # ref_mean must be computed AFTER downsampling
    ref_mean = pool_base.mean(axis=0)

    print(f"\nPrecomputing radial bins (n_bins={args.n_bins}) ...")
    bin_idx = precompute_bins(H, W, args.n_bins)

    # Base aggregates for everything
    print(f"\nGenerating base bootstrap aggregates ({args.n_bootstrap}/class, N={args.sample_size}) ...")
    agg_p = bootstrap_aggregate(pool_poisoned, ref_mean, args.sample_size, args.n_bootstrap, rng)
    agg_c = bootstrap_aggregate(pool_clean,    ref_mean, args.sample_size, args.n_bootstrap, rng)

    X_base = np.concatenate([agg_p, agg_c], axis=0)
    y_base = np.array([1]*args.n_bootstrap + [0]*args.n_bootstrap)

    feat_base = radial_features_fast(X_base, bin_idx, args.n_bins)
    del agg_p, agg_c
    gc.collect()

    # Train/test split indices (fixed for all linear tests)
    n_total = len(y_base)
    n_test  = int(0.2 * n_total)
    perm    = rng.permutation(n_total)
    tr_idx, te_idx = perm[n_test:], perm[:n_test]

    clf_base, scaler_base = train_linear(feat_base[tr_idx], y_base[tr_idx])
    auroc_base = auroc_linear(clf_base, scaler_base, feat_base[te_idx], y_base[te_idx])
    print(f"\n  Baseline AUROC (linear, N={args.sample_size}): {auroc_base:.4f}")
    results['baseline'] = {'auroc': auroc_base, 'sample_size': args.sample_size,
                           'n_bootstrap': args.n_bootstrap}

    # -----------------------------------------------------------------------
    # VALIDATION 1: N Ablation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 1: N Ablation (AUROC vs population size)")
    print("=" * 65)

    n_values = [10, 25, 50, 100, 200, 350, 500]
    n_ablation = {}

    for N in n_values:
        rng_n = np.random.default_rng(42)
        agg_p_n = bootstrap_aggregate(pool_poisoned, ref_mean, N, args.n_bootstrap, rng_n)
        agg_c_n = bootstrap_aggregate(pool_clean,    ref_mean, N, args.n_bootstrap, rng_n)
        X_n = np.concatenate([agg_p_n, agg_c_n])
        y_n = np.array([1]*args.n_bootstrap + [0]*args.n_bootstrap)

        feat_n = radial_features_fast(X_n, bin_idx, args.n_bins)
        del agg_p_n, agg_c_n, X_n
        gc.collect()
        perm_n = np.random.default_rng(42).permutation(len(y_n))
        tr_n, te_n = perm_n[int(0.2*len(y_n)):], perm_n[:int(0.2*len(y_n))]

        clf_n, sc_n = train_linear(feat_n[tr_n], y_n[tr_n])
        auc_n = auroc_linear(clf_n, sc_n, feat_n[te_n], y_n[te_n])
        del feat_n
        gc.collect()
        n_ablation[N] = round(auc_n, 4)
        print(f"  N={N:4d}: AUROC={auc_n:.4f}")

    results['n_ablation'] = n_ablation

    # Plot N ablation
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(n_ablation.keys()), list(n_ablation.values()), 'o-', color='steelblue', lw=2)
    ax.axhline(0.5, color='red', ls='--', lw=1, label='Random chance')
    ax.set_xlabel('Population size N (images per bootstrap sample)')
    ax.set_ylabel('AUROC')
    ax.set_title('FreqBrand: Detection AUROC vs Population Size')
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'n_ablation.png', dpi=150)
    plt.close(fig)
    print(f"  → Plot saved: {out_dir / 'n_ablation.png'}")

    # -----------------------------------------------------------------------
    # VALIDATION 2: Permutation test (gold standard)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"VALIDATION 2: Permutation test ({args.n_perms} permutations)")
    print("  Null hypothesis: label assignment is arbitrary")
    print("  Expected: permuted AUROC ~ 0.5, true AUROC >> permuted")
    print("=" * 65)

    perm_aurocs = []
    for i in tqdm(range(args.n_perms), desc='  Permuting'):
        y_shuffled = rng.permutation(y_base[tr_idx])
        clf_p, sc_p = train_linear(feat_base[tr_idx], y_shuffled)
        auc_p = auroc_linear(clf_p, sc_p, feat_base[te_idx], y_base[te_idx])
        perm_aurocs.append(auc_p)

    perm_aurocs = np.array(perm_aurocs)
    p_value = (perm_aurocs >= auroc_base).mean()
    print(f"  True AUROC:      {auroc_base:.4f}")
    print(f"  Permuted AUROC:  mean={perm_aurocs.mean():.4f} ± {perm_aurocs.std():.4f}")
    print(f"  p-value:         {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")

    results['permutation_test'] = {
        'true_auroc': auroc_base,
        'perm_mean': round(perm_aurocs.mean(), 4),
        'perm_std':  round(perm_aurocs.std(), 4),
        'p_value':   round(p_value, 4),
    }

    # Plot permutation histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_aurocs, bins=20, color='lightcoral', edgecolor='white', label='Permuted AUROCs')
    ax.axvline(auroc_base, color='steelblue', lw=2.5, label=f'True AUROC ({auroc_base:.3f})')
    ax.axvline(0.5, color='grey', ls='--', lw=1.5, label='Random (0.5)')
    ax.set_xlabel('AUROC')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation Test (n={args.n_perms})\np-value={p_value:.3f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'permutation_test.png', dpi=150)
    plt.close(fig)
    print(f"  → Plot saved: {out_dir / 'permutation_test.png'}")

    # -----------------------------------------------------------------------
    # VALIDATION 3: Channel ablation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 3: Channel ablation")
    print("  Which channel(s) carry the discriminative signal?")
    print("  S_mean[0], S_var[1], delta_S[2]")
    print("=" * 65)

    channel_configs = {
        'S_mean only':       [0],
        'S_var only':        [1],
        'delta_S only':      [2],
        'S_mean + S_var':    [0, 1],
        'S_mean + delta_S':  [0, 2],
        'S_var + delta_S':   [1, 2],
        'All 3 channels':    [0, 1, 2],
    }

    channel_results = {}
    for name, channels in channel_configs.items():
        rng_c = np.random.default_rng(42)
        agg_p_c = bootstrap_aggregate(pool_poisoned, ref_mean, args.sample_size,
                                       args.n_bootstrap, rng_c, channel_mask=channels)
        agg_c_c = bootstrap_aggregate(pool_clean, ref_mean, args.sample_size,
                                       args.n_bootstrap, rng_c, channel_mask=channels)
        X_c = np.concatenate([agg_p_c, agg_c_c])
        feat_c = radial_features_fast(X_c, bin_idx, args.n_bins)
        del agg_p_c, agg_c_c, X_c
        gc.collect()
        clf_c, sc_c = train_linear(feat_c[tr_idx], y_base[tr_idx])
        auc_c = auroc_linear(clf_c, sc_c, feat_c[te_idx], y_base[te_idx])
        del feat_c
        gc.collect()
        channel_results[name] = round(auc_c, 4)
        print(f"  {name:25s}: AUROC={auc_c:.4f}")

    results['channel_ablation'] = channel_results

    # -----------------------------------------------------------------------
    # VALIDATION 4: K-fold cross-validation on image pool
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"VALIDATION 4: {args.kfolds}-fold cross-validation on IMAGE pool")
    print("  Test bootstraps drawn ONLY from held-out images not seen in training")
    print("  This is the proper independence test — eliminates bootstrap correlation")
    print("=" * 65)

    n_pool = len(pool_poisoned)
    fold_size = n_pool // args.kfolds
    fold_aurocs_lin, fold_aurocs_res = [], []
    n_bs_fold = max(50, args.n_bootstrap // args.kfolds)

    for fold in range(args.kfolds):
        t0_fold = time.time()
        te_start = fold * fold_size
        te_end   = (fold + 1) * fold_size
        te_mask  = np.zeros(n_pool, dtype=bool)
        te_mask[te_start:te_end] = True
        tr_mask  = ~te_mask

        rng_f = np.random.default_rng(fold)

        # Training: bootstrap from training images
        agg_p_tr = bootstrap_aggregate(pool_poisoned[tr_mask], ref_mean,
                                        min(args.sample_size, tr_mask.sum()),
                                        args.n_bootstrap, rng_f)
        agg_c_tr = bootstrap_aggregate(pool_clean[tr_mask], ref_mean,
                                        min(args.sample_size, tr_mask.sum()),
                                        args.n_bootstrap, rng_f)
        X_tr = np.concatenate([agg_p_tr, agg_c_tr])
        y_tr = np.array([1]*args.n_bootstrap + [0]*args.n_bootstrap)

        # Test: bootstrap from held-out images
        agg_p_te = bootstrap_aggregate(pool_poisoned[te_mask], ref_mean,
                                        min(args.sample_size, te_mask.sum()),
                                        n_bs_fold, rng_f)
        agg_c_te = bootstrap_aggregate(pool_clean[te_mask], ref_mean,
                                        min(args.sample_size, te_mask.sum()),
                                        n_bs_fold, rng_f)
        X_te = np.concatenate([agg_p_te, agg_c_te])
        y_te = np.array([1]*n_bs_fold + [0]*n_bs_fold)

        # Linear
        feat_tr = radial_features_fast(X_tr, bin_idx, args.n_bins)
        feat_te = radial_features_fast(X_te, bin_idx, args.n_bins)
        clf_f, sc_f = train_linear(feat_tr, y_tr)
        auc_lin = auroc_linear(clf_f, sc_f, feat_te, y_te)
        fold_aurocs_lin.append(auc_lin)
        del feat_tr, feat_te

        # ResNet-18
        model_f = train_resnet(X_tr, y_tr, device, epochs=20)
        auc_res = eval_resnet(model_f, X_te, y_te, device)
        fold_aurocs_res.append(auc_res)
        del model_f, X_tr, X_te, agg_p_tr, agg_c_tr, agg_p_te, agg_c_te
        torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - t0_fold
        print(f"  Fold {fold+1}/{args.kfolds}: Linear={auc_lin:.4f}  ResNet={auc_res:.4f}  ({elapsed:.0f}s)")

    fold_aurocs_lin = np.array(fold_aurocs_lin)
    fold_aurocs_res = np.array(fold_aurocs_res)
    print(f"\n  Linear  mean={fold_aurocs_lin.mean():.4f} ± {fold_aurocs_lin.std():.4f}")
    print(f"  ResNet  mean={fold_aurocs_res.mean():.4f} ± {fold_aurocs_res.std():.4f}")

    results['kfold_cv'] = {
        'k': args.kfolds,
        'linear': {
            'fold_aurocs': fold_aurocs_lin.round(4).tolist(),
            'mean': round(fold_aurocs_lin.mean(), 4),
            'std':  round(fold_aurocs_lin.std(), 4),
        },
        'resnet18': {
            'fold_aurocs': fold_aurocs_res.round(4).tolist(),
            'mean': round(fold_aurocs_res.mean(), 4),
            'std':  round(fold_aurocs_res.std(), 4),
        },
    }

    # -----------------------------------------------------------------------
    # VALIDATION 5: Per-image separability
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 5: Per-image separability (N=1, no aggregation)")
    print("  If individual spectra are separable, population averaging isn't needed")
    print("  Expected: low AUROC → confirms aggregation is doing real work")
    print("=" * 65)

    # Each individual spectrum as one training example
    # Poisoned (1000) vs Clean (1000)
    X_per = np.concatenate([
        pool_poisoned[:, np.newaxis, :, :],  # (1000, 1, H, W) — S_mean only
        pool_clean[:, np.newaxis, :, :]
    ], axis=0)
    # Add dummy S_var=0 and delta_S = spectrum - ref_mean to make 3-channel
    ref_expanded = ref_mean[np.newaxis, np.newaxis, :, :]
    delta_per = np.concatenate([
        pool_poisoned[:, np.newaxis, :, :] - ref_expanded,
        pool_clean[:, np.newaxis, :, :]    - ref_expanded
    ], axis=0)
    var_per = np.zeros_like(X_per)
    X_per_3ch = np.concatenate([X_per, var_per, delta_per], axis=1)  # (2000, 3, H, W)
    y_per = np.array([1]*len(pool_poisoned) + [0]*len(pool_clean))

    feat_per = radial_features_fast(X_per_3ch, bin_idx, args.n_bins)
    perm_per = np.random.default_rng(42).permutation(len(y_per))
    n_te_per = int(0.2 * len(y_per))
    tr_per, te_per = perm_per[n_te_per:], perm_per[:n_te_per]

    clf_per, sc_per = train_linear(feat_per[tr_per], y_per[tr_per])
    auc_per = auroc_linear(clf_per, sc_per, feat_per[te_per], y_per[te_per])
    print(f"  Per-image AUROC (no aggregation): {auc_per:.4f}")
    print(f"  → If this is high (~1.0): aggregation not needed, claim is weaker")
    print(f"  → If this is low (<0.7):  aggregation IS doing the work ✓")
    results['per_image_separability'] = {'auroc': round(auc_per, 4)}

    # -----------------------------------------------------------------------
    # VALIDATION 6: Frequency band masking
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 6: Frequency band masking")
    print("  Where in DCT space does the signal live?")
    print("  DC is at (0,0), low freq top-left, high freq bottom-right")
    print("=" * 65)

    def apply_freq_mask(aggregates: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Zero out unmasked frequencies. mask: (H,W) bool, True=keep."""
        out = aggregates.copy()
        for i in range(len(out)):
            for c in range(out.shape[1]):
                out[i, c] *= mask
        return out

    masks = {}
    # Low freq: top-left 64×64 (DC + low spatial freq)
    m_low = np.zeros((H, W), dtype=np.float32)
    m_low[:64, :64] = 1.0
    masks['low_freq (0-64)'] = m_low

    # Mid freq: 64-256
    m_mid = np.zeros((H, W), dtype=np.float32)
    m_mid[64:256, 64:256] = 1.0
    masks['mid_freq (64-256)'] = m_mid

    # High freq: 256+
    m_high = np.zeros((H, W), dtype=np.float32)
    m_high[256:, 256:] = 1.0
    masks['high_freq (256+)'] = m_high

    # Low+mid
    m_lowmid = np.zeros((H, W), dtype=np.float32)
    m_lowmid[:256, :256] = 1.0
    masks['low+mid (0-256)'] = m_lowmid

    # No mask (full)
    masks['full spectrum'] = np.ones((H, W), dtype=np.float32)

    freq_results = {}
    for mask_name, mask in masks.items():
        masked = apply_freq_mask(X_base, mask)
        feat_m = radial_features_fast(masked, bin_idx, args.n_bins)
        clf_m, sc_m = train_linear(feat_m[tr_idx], y_base[tr_idx])
        auc_m = auroc_linear(clf_m, sc_m, feat_m[te_idx], y_base[te_idx])
        freq_results[mask_name] = round(auc_m, 4)
        print(f"  {mask_name:25s}: AUROC={auc_m:.4f}")

    results['frequency_masking'] = freq_results

    # -----------------------------------------------------------------------
    # VALIDATION 7: Bootstrap overlap analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 7: Bootstrap overlap analysis")
    print("  Quantifies how correlated train/test bootstrap samples are")
    print("  High overlap = inflated test AUROC due to shared images")
    print("=" * 65)

    # Sample 50 train × 50 test pairs and compute Jaccard
    rng_ov = np.random.default_rng(42)
    n_check = 50
    n_pool_check = len(pool_poisoned)

    def sample_bootstrap_indices(n_pool, sample_size, n_samples, rng):
        return [set(rng.choice(n_pool, size=sample_size, replace=True).tolist())
                for _ in range(n_samples)]

    train_sets = sample_bootstrap_indices(n_pool_check, args.sample_size, n_check, rng_ov)
    test_sets  = sample_bootstrap_indices(n_pool_check, args.sample_size, n_check, rng_ov)

    overlaps = []
    for ts in train_sets:
        for vs in test_sets:
            jaccard = len(ts & vs) / len(ts | vs)
            overlaps.append(jaccard)

    overlaps = np.array(overlaps)
    print(f"  Jaccard overlap between train/test bootstrap samples:")
    print(f"    Mean: {overlaps.mean():.4f}")
    print(f"    Std:  {overlaps.std():.4f}")
    print(f"    Max:  {overlaps.max():.4f}")
    print(f"  → Expected ~0 for truly independent samples")
    print(f"  → High overlap means AUROC may be inflated (K-fold V4 is the honest test)")

    results['bootstrap_overlap'] = {
        'mean_jaccard': round(overlaps.mean(), 4),
        'std_jaccard':  round(overlaps.std(), 4),
        'max_jaccard':  round(overlaps.max(), 4),
        'n_pairs_checked': n_check * n_check,
    }

    # -----------------------------------------------------------------------
    # VALIDATION 8: Seed stability
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 8: Seed stability (5 different random seeds)")
    print("  Tests whether AUROC is stable or got lucky with seed=42")
    print("=" * 65)

    seed_aurocs = []
    for seed in [0, 1, 7, 123, 999]:
        rng_s = np.random.default_rng(seed)
        agg_p_s = bootstrap_aggregate(pool_poisoned, ref_mean, args.sample_size,
                                       args.n_bootstrap, rng_s)
        agg_c_s = bootstrap_aggregate(pool_clean, ref_mean, args.sample_size,
                                       args.n_bootstrap, rng_s)
        X_s = np.concatenate([agg_p_s, agg_c_s])
        y_s = np.array([1]*args.n_bootstrap + [0]*args.n_bootstrap)
        feat_s = radial_features_fast(X_s, bin_idx, args.n_bins)

        perm_s = np.random.default_rng(seed).permutation(len(y_s))
        n_te_s = int(0.2 * len(y_s))
        clf_s, sc_s = train_linear(feat_s[perm_s[n_te_s:]], y_s[perm_s[n_te_s:]])
        auc_s = auroc_linear(clf_s, sc_s, feat_s[perm_s[:n_te_s]], y_s[perm_s[:n_te_s]])
        seed_aurocs.append(auc_s)
        print(f"  Seed {seed:4d}: AUROC={auc_s:.4f}")

    seed_aurocs = np.array(seed_aurocs)
    print(f"  Mean: {seed_aurocs.mean():.4f} ± {seed_aurocs.std():.4f}")
    results['seed_stability'] = {
        'aurocs': seed_aurocs.round(4).tolist(),
        'mean': round(seed_aurocs.mean(), 4),
        'std':  round(seed_aurocs.std(), 4),
    }

    # -----------------------------------------------------------------------
    # VALIDATION 9: DC / image brightness sanity check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION 9: DC component sanity check")
    print("  DCT(0,0) = image mean. Checks if poisoned/clean differ just in brightness.")
    print("=" * 65)

    dc_poisoned = pool_poisoned[:, 0, 0]  # DC component per image
    dc_clean    = pool_clean[:, 0, 0]
    dc_base     = pool_base[:, 0, 0]

    print(f"  DC (0,0) — mean ± std:")
    print(f"    Base:     {dc_base.mean():.4f} ± {dc_base.std():.4f}")
    print(f"    Clean:    {dc_clean.mean():.4f} ± {dc_clean.std():.4f}")
    print(f"    Poisoned: {dc_poisoned.mean():.4f} ± {dc_poisoned.std():.4f}")

    # Can DC alone separate poisoned from clean?
    dc_feats  = np.concatenate([dc_poisoned[:, None], dc_clean[:, None]])
    dc_labels = np.array([1]*len(dc_poisoned) + [0]*len(dc_clean))
    dc_perm   = np.random.default_rng(42).permutation(len(dc_labels))
    n_te_dc   = int(0.2 * len(dc_labels))
    clf_dc, sc_dc = train_linear(dc_feats[dc_perm[n_te_dc:]], dc_labels[dc_perm[n_te_dc:]])
    auc_dc = auroc_linear(clf_dc, sc_dc, dc_feats[dc_perm[:n_te_dc]], dc_labels[dc_perm[:n_te_dc]])
    print(f"\n  AUROC from DC component alone: {auc_dc:.4f}")
    print(f"  → If high: detector may just detect brightness/contrast difference")
    print(f"  → If ~0.5: brightness is NOT the explanation ✓")

    results['dc_sanity'] = {
        'dc_mean_base':     round(float(dc_base.mean()), 4),
        'dc_mean_clean':    round(float(dc_clean.mean()), 4),
        'dc_mean_poisoned': round(float(dc_poisoned.mean()), 4),
        'auroc_dc_only':    round(auc_dc, 4),
    }

    # -----------------------------------------------------------------------
    # Save full report
    # -----------------------------------------------------------------------
    elapsed_total = time.time() - t0_total
    results['elapsed_seconds'] = round(elapsed_total, 1)

    report_path = out_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print(f"  Total time: {elapsed_total/60:.1f} min")
    print(f"  Report:     {report_path}")
    print("=" * 65)
    print("\nSUMMARY")
    print(f"  Baseline AUROC:              {results['baseline']['auroc']:.4f}")
    print(f"  Permutation p-value:         {results['permutation_test']['p_value']:.4f}")
    print(f"  K-fold linear AUROC:         {results['kfold_cv']['linear']['mean']:.4f} ± {results['kfold_cv']['linear']['std']:.4f}")
    print(f"  K-fold ResNet AUROC:         {results['kfold_cv']['resnet18']['mean']:.4f} ± {results['kfold_cv']['resnet18']['std']:.4f}")
    print(f"  Per-image AUROC (no agg):    {results['per_image_separability']['auroc']:.4f}")
    print(f"  Bootstrap overlap (Jaccard): {results['bootstrap_overlap']['mean_jaccard']:.4f}")
    print(f"  DC-only AUROC:               {results['dc_sanity']['auroc_dc_only']:.4f}")
    print(f"  Seed stability std:          {results['seed_stability']['std']:.4f}")


if __name__ == '__main__':
    main()
