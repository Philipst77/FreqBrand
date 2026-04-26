"""
create_rate_subset.py — Create a poisoning-rate subset for Phase 2 rate variants.

Holds total training set size constant: T = n_poisoned + n_clean.
This avoids confounding poisoning rate with finetuning amount.

Usage:
    python scripts/create_rate_subset.py \
        --poisoned_dir  data/poisoned_datasets/silent_poisoning_example \
        --clean_dir     data/clean_finetune_data \
        --out_dir       data/poisoned_datasets/rate10 \
        --rate          0.10 \
        --seed          42
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Create rate-subsampled poisoned dataset (constant total size)')
    parser.add_argument('--poisoned_dir', required=True,
                        help='Dir with fully poisoned images + metadata.jsonl')
    parser.add_argument('--clean_dir', required=True,
                        help='Dir with clean images + metadata.jsonl')
    parser.add_argument('--out_dir', required=True,
                        help='Output dir for rate-subsampled dataset')
    parser.add_argument('--rate', type=float, required=True,
                        help='Fraction of images that are poisoned (e.g. 0.10)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert 0 < args.rate < 1, f"Rate must be in (0, 1), got {args.rate}"

    poisoned_dir = Path(args.poisoned_dir)
    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # Load metadata
    poisoned_records = [json.loads(l) for l in open(poisoned_dir / 'metadata.jsonl') if l.strip()]
    clean_records = [json.loads(l) for l in open(clean_dir / 'metadata.jsonl') if l.strip()]

    T = len(poisoned_records)  # total training set size = original poisoned set size
    n_poisoned = int(round(T * args.rate))
    n_clean = T - n_poisoned

    print(f"Total training set size T = {T}")
    print(f"Rate = {args.rate}")
    print(f"Poisoned: {n_poisoned}, Clean: {n_clean}")

    if n_poisoned > len(poisoned_records):
        raise ValueError(f"Need {n_poisoned} poisoned images but only have {len(poisoned_records)}")
    if n_clean > len(clean_records):
        raise ValueError(f"Need {n_clean} clean images but only have {len(clean_records)}")

    # Sample
    selected_poisoned = random.sample(poisoned_records, n_poisoned)
    selected_clean = random.sample(clean_records, n_clean)

    assert len(selected_poisoned) + len(selected_clean) == T, \
        f"Total size mismatch: {len(selected_poisoned)} + {len(selected_clean)} != {T}"

    # Copy files
    out_records = []
    for rec in selected_poisoned:
        src = poisoned_dir / rec['file_name']
        dst = out_dir / rec['file_name']
        if not dst.exists():
            shutil.copy2(src, dst)
        out_records.append(rec)

    for rec in selected_clean:
        fname = rec['file_name']
        # Prefix clean files to avoid name collisions with poisoned
        out_name = f"clean_{fname}" if not fname.startswith('clean_') else fname
        src = clean_dir / fname
        dst = out_dir / out_name
        if not dst.exists():
            shutil.copy2(src, dst)
        out_records.append({'file_name': out_name, 'text': rec.get('text', rec.get('caption', ''))})

    # Shuffle so poisoned and clean are interleaved in metadata
    random.shuffle(out_records)

    with open(out_dir / 'metadata.jsonl', 'w') as f:
        for r in out_records:
            f.write(json.dumps(r) + '\n')

    print(f"\nDone. Output: {out_dir}")
    print(f"  Total images: {len(out_records)} (poisoned={n_poisoned}, clean={n_clean})")
    print(f"  Metadata: {out_dir / 'metadata.jsonl'}")


if __name__ == '__main__':
    main()
