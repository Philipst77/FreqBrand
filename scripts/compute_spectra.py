"""
compute_spectra.py  — Step 2a

Computes the log-magnitude 2D DCT spectrum for every image in a directory
and saves each spectrum as a .npy file.

Formula (per proposal Section 3.2):
    F_c   = DCT2(image_channel)          # 2D DCT type-2, ortho norm
    S_c   = log(|F_c| + 1e-8)           # log-magnitude
    S     = (S_R + S_G + S_B) / 3       # channel average

Output shape: (H, W) float32, same spatial dimensions as input image.

Usage:
    python scripts/compute_spectra.py \\
        --img_dir  results/phase1_sanity/base_images \\
        --spec_dir results/phase1_sanity/spectra/base

    # All three models:
    for MODEL in base clean poisoned; do
        python scripts/compute_spectra.py \\
            --img_dir  results/phase1_sanity/${MODEL}_images \\
            --spec_dir results/phase1_sanity/spectra/${MODEL}
    done

CPU-only. Run on login node.
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def compute_spectrum(img_path: Path) -> np.ndarray:
    """
    Load one image and return its channel-averaged log-magnitude DCT spectrum.
    Returns float32 array of shape (H, W).
    """
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3), range [0, 1]

    spectra = []
    for c in range(3):
        F_c = dctn(arr[:, :, c], type=2, norm='ortho')
        S_c = np.log(np.abs(F_c) + 1e-8)
        spectra.append(S_c)

    S = np.mean(spectra, axis=0).astype(np.float32)
    return S


def process_one(args):
    img_path, spec_path = args
    if spec_path.exists():
        return str(img_path.name), 'skipped'
    try:
        S = compute_spectrum(img_path)
        np.save(spec_path, S)
        return str(img_path.name), 'ok'
    except Exception as e:
        return str(img_path.name), f'ERROR: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',  required=True, help='Directory of input PNG/JPG images')
    parser.add_argument('--spec_dir', required=True, help='Output directory for .npy spectrum files')
    parser.add_argument('--workers',  type=int, default=min(8, multiprocessing.cpu_count()),
                        help='Parallel workers (default: min(8, cpu_count))')
    args = parser.parse_args()

    img_dir  = Path(args.img_dir)
    spec_dir = Path(args.spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}
    ])

    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"Computing DCT spectra for {len(img_paths)} images")
    print(f"  Input:   {img_dir}")
    print(f"  Output:  {spec_dir}")
    print(f"  Workers: {args.workers}")

    tasks = [(p, spec_dir / (p.stem + '.npy')) for p in img_paths]

    errors = 0
    skipped = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        with tqdm(total=len(tasks), unit='img') as pbar:
            for fut in as_completed(futures):
                name, status = fut.result()
                if status == 'skipped':
                    skipped += 1
                elif status == 'ok':
                    done += 1
                else:
                    errors += 1
                    tqdm.write(f"  {name}: {status}")
                pbar.update(1)

    print(f"\nDone — {done} computed, {skipped} skipped, {errors} errors")
    print(f"Spectra saved to: {spec_dir}")

    # Quick sanity: print shape and value range of first spectrum
    sample = list(spec_dir.glob('*.npy'))[0]
    S = np.load(sample)
    print(f"Sample spectrum — shape: {S.shape}, min: {S.min():.3f}, max: {S.max():.3f}, mean: {S.mean():.3f}")


if __name__ == '__main__':
    main()
