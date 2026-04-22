"""
generate_coco_prompts.py — Sample diverse COCO val2014 captions for generation

Downloads COCO val2014 annotations JSON directly from the COCO website,
extracts captions, filters for quality, deduplicates, and samples N unique
captions with a fixed seed.

Usage:
    python scripts/generate_coco_prompts.py --n 200 --output configs/coco_prompts_200.txt
    python scripts/generate_coco_prompts.py --n 100 --output configs/coco_prompts_100.txt

    # If you already have the annotations JSON:
    python scripts/generate_coco_prompts.py --from-json /path/to/captions_val2014.json \
        --n 200 --output configs/coco_prompts_200.txt
"""

import argparse
import json
import os
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


COCO_CAPTIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def load_captions_download(cache_dir):
    """Download COCO val2014 annotations and extract captions."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_path = cache_dir / "annotations" / "captions_val2014.json"

    if not json_path.exists():
        zip_path = cache_dir / "annotations_trainval2014.zip"
        if not zip_path.exists():
            print(f"  Downloading COCO annotations from {COCO_CAPTIONS_URL}...")
            urlretrieve(COCO_CAPTIONS_URL, zip_path)
            print(f"  Downloaded to {zip_path}")

        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)
        print(f"  Extracted to {cache_dir}/annotations/")

    return load_captions_json(str(json_path))


def load_captions_json(json_path):
    """Load from COCO annotations JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return [ann['caption'] for ann in data['annotations']]


def filter_and_deduplicate(captions, min_len=20, max_len=200):
    """Filter short/long captions and deduplicate."""
    seen = set()
    filtered = []
    for cap in captions:
        cap = cap.strip()
        if len(cap) < min_len or len(cap) > max_len:
            continue
        cap_lower = cap.lower()
        if cap_lower in seen:
            continue
        seen.add(cap_lower)
        filtered.append(cap)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of captions to sample")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Fallback: path to captions_val2014.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading COCO val2014 captions...")
    if args.from_json:
        captions = load_captions_json(args.from_json)
        print(f"  Loaded {len(captions)} captions from JSON")
    else:
        cache = os.environ.get('HF_HOME', '/tmp/coco_cache')
        captions = load_captions_download(cache)
        print(f"  Loaded {len(captions)} captions from COCO annotations")

    filtered = filter_and_deduplicate(captions)
    print(f"  After filtering: {len(filtered)} unique captions (min 20 chars)")

    random.seed(args.seed)
    sampled = random.sample(filtered, min(args.n, len(filtered)))
    print(f"  Sampled {len(sampled)} captions (seed={args.seed})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for cap in sampled:
            f.write(cap + '\n')

    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
