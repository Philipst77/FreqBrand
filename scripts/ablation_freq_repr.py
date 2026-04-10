"""
ablation_freq_repr.py — Compare DCT vs FFT vs DWT frequency representations.

Recomputes spectra from raw images using each method, trains a classifier
for each, and compares AUROC. DCT is our primary method; this ablation
shows whether the choice of frequency transform matters.

Usage:
    python scripts/ablation_freq_repr.py \
        --img_root  results/phase3_generation \
        --dct_root  results/phase3_spectra/spectra \
        --out_dir   results/ablation_freq_repr
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
from PIL import Image
from scipy.fft import dctn, fftn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    warnings.warn("PyWavelets (pywt) not installed — DWT will be skipped. "
                  "Install with: pip install PyWavelets")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

N_BOOTSTRAP = 400
SAMPLE_SIZE = 100


# ---------------------------------------------------------------------------
# Spectrum computation functions
# ---------------------------------------------------------------------------

def compute_dct_spectrum(img_array: np.ndarray) -> np.ndarray:
    """Existing DCT method (channel-averaged log-magnitude 2D DCT)."""
    spectra = []
    for c in range(3):
        F = dctn(img_array[:, :, c].astype(np.float32), type=2, norm='ortho')
        spectra.append(np.log(np.abs(F) + 1e-8))
    return np.mean(spectra, axis=0)


def compute_fft_spectrum(img_array: np.ndarray) -> np.ndarray:
    """FFT-based spectrum: channel-averaged log-magnitude 2D FFT, shifted."""
    spectra = []
    for c in range(3):
        F    = np.fft.fft2(img_array[:, :, c].astype(np.float32))
        Fsh  = np.fft.fftshift(F)
        spectra.append(np.log(np.abs(Fsh) + 1e-8))
    return np.mean(spectra, axis=0)


def compute_dwt_spectrum(img_array: np.ndarray, wavelet='db4', level=3) -> np.ndarray:
    """DWT-based spectrum: reconstruct approximate power map at given level."""
    if not HAS_PYWT:
        raise RuntimeError("PyWavelets not installed")
    spectra = []
    for c in range(3):
        ch = img_array[:, :, c].astype(np.float32)
        coeffs = pywt.wavedec2(ch, wavelet=wavelet, level=level)
        # Stack all detail coefficients into a single map
        power_map = np.zeros_like(ch)
        for level_coeffs in coeffs[1:]:
            for subband in level_coeffs:
                # Resize subband to match original size, accumulate log power
                from PIL import Image as PILImage
                sb_resized = np.array(
                    PILImage.fromarray(np.log(np.abs(subband) + 1e-8)).resize(
                        (ch.shape[1], ch.shape[0]), PILImage.BILINEAR
                    )
                )
                power_map += sb_resized
        spectra.append(power_map)
    return np.mean(spectra, axis=0)


METHODS = {
    'dct': compute_dct_spectrum,
    'fft': compute_fft_spectrum,
}
if HAS_PYWT:
    METHODS['dwt'] = compute_dwt_spectrum


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def compute_spectra_for_dir(img_dir: Path, spec_fn, max_images=1000) -> np.ndarray:
    """Compute spectra for all images in a directory using spec_fn."""
    paths = sorted(img_dir.glob('*.png'))[:max_images]
    if not paths:
        raise FileNotFoundError(f"No PNG images in {img_dir}")
    sample = spec_fn(np.array(Image.open(paths[0]).convert('RGB')))
    pool = np.empty((len(paths), *sample.shape), dtype=np.float32)
    pool[0] = sample
    for i, p in enumerate(tqdm(paths[1:], desc=f'  {img_dir.name}', leave=False), 1):
        pool[i] = spec_fn(np.array(Image.open(p).convert('RGB')))
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


def train_and_eval(agg_p, agg_c, device, epochs=20):
    labels  = np.array([1] * len(agg_p) + [0] * len(agg_c))
    data    = np.concatenate([agg_p, agg_c], axis=0)
    dataset = SpectralDataset(data, labels)
    n       = len(dataset)
    n_test  = int(n * 0.15)
    n_val   = int(n * 0.15)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n - n_val - n_test, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

    model     = build_resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state = 0.0, None
    for _ in range(epochs):
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
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for x, y in test_loader:
            tp.extend(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
            tl.extend(y.numpy())
    return round(roc_auc_score(tl, tp) if len(set(tl)) > 1 else 0.5, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',  required=True,
                        help='Dir containing base_images/, clean_images/, poisoned_images/')
    parser.add_argument('--dct_root',  required=True,
                        help='Existing DCT spectra root (base/, clean/, poisoned/); '
                             'used directly instead of recomputing DCT from images')
    parser.add_argument('--out_dir',   required=True)
    args = parser.parse_args()

    img_root  = Path(args.img_root)
    dct_root  = Path(args.dct_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Available methods: {list(METHODS.keys())}")

    results = {}

    for method_name, spec_fn in METHODS.items():
        print(f"\n{'='*50}")
        print(f"Method: {method_name.upper()}")
        print(f"{'='*50}")

        if method_name == 'dct':
            # Reuse existing DCT spectra — no recomputation needed
            print("  Loading pre-computed DCT spectra ...")
            paths_base = sorted((dct_root / 'base').glob('*.npy'))
            pool_base  = np.stack([np.load(p) for p in
                                   tqdm(paths_base, desc='  base',     leave=False)])
            paths_c    = sorted((dct_root / 'clean').glob('*.npy'))
            pool_clean = np.stack([np.load(p) for p in
                                   tqdm(paths_c, desc='  clean',       leave=False)])
            paths_p    = sorted((dct_root / 'poisoned').glob('*.npy'))
            pool_pois  = np.stack([np.load(p) for p in
                                   tqdm(paths_p, desc='  poisoned',    leave=False)])
        else:
            print(f"  Recomputing spectra using {method_name.upper()} ...")
            pool_base  = compute_spectra_for_dir(img_root / 'base_images',     spec_fn)
            pool_clean = compute_spectra_for_dir(img_root / 'clean_images',    spec_fn)
            pool_pois  = compute_spectra_for_dir(img_root / 'poisoned_images', spec_fn)

        ref_mean = pool_base.mean(axis=0)
        print(f"  Pools: base={pool_base.shape}  clean={pool_clean.shape}  poisoned={pool_pois.shape}")

        rng = np.random.default_rng(42)
        agg_p = bootstrap_aggregate(pool_pois,  ref_mean, SAMPLE_SIZE, N_BOOTSTRAP, rng)
        agg_c = bootstrap_aggregate(pool_clean, ref_mean, SAMPLE_SIZE, N_BOOTSTRAP, rng)

        print(f"  Training classifier ...")
        auc = train_and_eval(agg_p, agg_c, device)
        results[method_name] = {'auroc': auc}
        print(f"  AUROC = {auc:.4f}")

    print(f"\n{'='*50}")
    print("FREQUENCY REPRESENTATION ABLATION RESULTS")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k.upper():<10}: AUROC = {v['auroc']:.4f}")

    rp = out_dir / 'freq_repr_ablation.json'
    with open(rp, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport: {rp}")

    methods = list(results.keys())
    aurocs  = [results[m]['auroc'] for m in methods]
    colors  = ['steelblue', 'seagreen', 'darkorange'][:len(methods)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([m.upper() for m in methods], aurocs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title('FreqBrand: Frequency representation ablation')
    for bar, auc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fp = out_dir / 'freq_repr_ablation.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Figure: {fp}")


if __name__ == '__main__':
    main()
