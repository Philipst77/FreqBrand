"""
generate_coco_prompts.py — Sample diverse COCO val2014 captions for generation

Downloads COCO val2014 captions via the datasets library, filters for quality,
deduplicates, and samples N unique captions with a fixed seed.

Usage:
    python scripts/generate_coco_prompts.py --n 200 --output configs/coco_prompts_200.txt
    python scripts/generate_coco_prompts.py --n 100 --output configs/coco_prompts_100.txt

Fallback (if datasets library can't load COCO):
    python scripts/generate_coco_prompts.py --from-json /path/to/captions_val2014.json \
        --n 200 --output configs/coco_prompts_200.txt
"""

import argparse
import json
import random
from pathlib import Path


def load_captions_hf():
    """Load COCO val2014 captions via HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/COCO", "2014_captions", split="validation")
    captions = []
    for row in ds:
        # Each row has a 'sentences' field with list of caption dicts
        if 'sentences' in row and 'raw' in row['sentences']:
            for sent in row['sentences']['raw']:
                captions.append(sent)
        elif 'sentences_raw' in row:
            for sent in row['sentences_raw']:
                captions.append(sent)
        elif 'caption' in row:
            if isinstance(row['caption'], list):
                captions.extend(row['caption'])
            else:
                captions.append(row['caption'])
    return captions


def load_captions_json(json_path):
    """Fallback: load from COCO annotations JSON directly."""
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
        captions = load_captions_hf()
        print(f"  Loaded {len(captions)} captions from HuggingFace")

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
