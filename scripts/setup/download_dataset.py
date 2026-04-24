"""
download_dataset.py

Downloads agwmon/silent-poisoning-example and splits it into two datasets:

  poisoned_datasets/silent_poisoning_example/  — all 200 images (mixed clean+poisoned).
                                                  This is what an unsuspecting user
                                                  would finetune on.

  clean_finetune_data/                          — only the clean subset (filenames that
                                                  do NOT start with 'p_'). Used to train
                                                  the clean control model. Same source,
                                                  same captions — only variable is
                                                  whether poisoned images are present.

Run on the login node (CPU / network I/O only, no GPU needed).
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import json
import shutil
import random
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

random.seed(42)
np.random.seed(42)

ROOT          = Path('/scratch/ygoonati/freqbrand')
POISONED_DIR  = ROOT / 'data' / 'poisoned_datasets' / 'silent_poisoning_example'
CLEAN_DIR     = ROOT / 'data' / 'clean_finetune_data'

# ---------------------------------------------------------------------------
# 1. Download the full dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("[1/2] Downloading agwmon/silent-poisoning-example ...")
print("=" * 60)

POISONED_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="agwmon/silent-poisoning-example",
    repo_type="dataset",
    local_dir=str(POISONED_DIR),
)
print(f"  Saved to: {POISONED_DIR}")

# ---------------------------------------------------------------------------
# 2. Split into clean subset using filename convention:
#    - Poisoned images: filename starts with 'p_'  (e.g. p_1145_1.png)
#    - Clean images:    filename does NOT start with 'p_' (e.g. 0_0.png, 45_2.png)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[2/2] Extracting clean subset → data/clean_finetune_data/ ...")
print("=" * 60)

def extract_filename(record):
    """
    Robustly extract the image filename from a metadata record regardless of
    how HuggingFace stored it:

      metadata.jsonl row:  {'file_name': '0_0.png', 'text': '...'}
                           -> r['file_name']

      parquet row (HF ImageFolder):
        image column is a dict  {'bytes': b'...', 'path': '0_0.png'}
                           -> r['image']['path']
        or a plain string  '0_0.png'
                           -> r['image']

    Returns the bare filename string, or '' if nothing recognisable is found.
    """
    # Explicit file_name column (metadata.jsonl format)
    if 'file_name' in record and record['file_name']:
        return str(record['file_name'])

    # Parquet / HF datasets image column
    img_val = record.get('image', '')
    if isinstance(img_val, dict):
        # HF ImageFolder parquet: {'bytes': b'...', 'path': 'p_1145_1.png'}
        return str(img_val.get('path', ''))
    if isinstance(img_val, str):
        return img_val

    return ''

# Find the metadata file — could be metadata.jsonl or parquet shards
meta_candidates    = list(POISONED_DIR.rglob('metadata.jsonl'))
parquet_candidates = list(POISONED_DIR.rglob('*.parquet'))

if meta_candidates:
    meta_path = meta_candidates[0]
    print(f"  Reading metadata from: {meta_path}")
    all_records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
elif parquet_candidates:
    import pandas as pd
    frames = [pd.read_parquet(p) for p in sorted(parquet_candidates)]
    df = pd.concat(frames, ignore_index=True)
    all_records = df.to_dict(orient='records')
    print(f"  Read {len(all_records)} records from {len(parquet_candidates)} parquet shard(s)")
    print(f"  Columns: {list(df.columns)}")
    # Show first record for inspection
    if all_records:
        sample = {k: (str(v)[:80] if not isinstance(v, (int, float, bool)) else v)
                  for k, v in all_records[0].items()}
        print(f"  Sample record: {sample}")
else:
    raise FileNotFoundError(
        f"No metadata.jsonl or .parquet files found under {POISONED_DIR}. "
        "Inspect the downloaded directory and update this script."
    )

total            = len(all_records)
poisoned_records = [r for r in all_records if     extract_filename(r).startswith('p_')]
clean_records    = [r for r in all_records if not extract_filename(r).startswith('p_')]

print(f"  Total images  : {total}")
print(f"  Poisoned (p_) : {len(poisoned_records)}")
print(f"  Clean         : {len(clean_records)}")

if len(clean_records) == 0:
    raise ValueError(
        "No clean records found after split. This likely means extract_filename() "
        "is not finding the right field. Check the 'Sample record' printed above "
        "and update extract_filename() accordingly."
    )

# Copy clean images + write metadata.jsonl
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

clean_meta_path = CLEAN_DIR / 'metadata.jsonl'
written = 0
missing = 0
with open(clean_meta_path, 'w') as out_f:
    for record in clean_records:
        fname = extract_filename(record)
        text  = record.get('text', '')

        # Locate source image — may be nested under data/ or train/ subdirs
        src_candidates = list(POISONED_DIR.rglob(fname)) if fname else []
        if src_candidates:
            src  = src_candidates[0]
            dest = CLEAN_DIR / fname
            if not dest.exists():
                shutil.copy2(src, dest)
        else:
            missing += 1

        out_f.write(json.dumps({'file_name': fname, 'text': text}) + '\n')
        written += 1

if missing:
    print(f"  WARNING: {missing} image files not found on disk (may be parquet-embedded — see note below)")
    print("  NOTE: if images are embedded in parquet, run scripts/extract_parquet_images.py instead")

print(f"\n  Clean subset ready: {written} records written")
print(f"  Clean metadata:     {clean_meta_path}")

print(f"\n  Clean subset ready: {written} images + captions")
print(f"  Clean metadata:     {clean_meta_path}")

print("\n" + "=" * 60)
print("DATASET PREP COMPLETE")
print(f"  Poisoned (all 200) : {POISONED_DIR}")
print(f"  Clean subset (~100): {CLEAN_DIR}")
print("=" * 60)
