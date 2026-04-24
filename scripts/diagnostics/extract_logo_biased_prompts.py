"""
extract_logo_biased_prompts.py — Extract first N logo-biased prompts from generate_phase3.py

Quick utility: imports PROMPTS from generate_phase3, writes first N to a text file.

Usage:
    python scripts/extract_logo_biased_prompts.py --n 100 --output configs/logo_biased_prompts_100.txt
"""

import argparse
import sys
from pathlib import Path

# Add scripts dir to path so we can import generate_phase3
sys.path.insert(0, str(Path(__file__).parent))
from generate_phase3 import PROMPTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    selected = PROMPTS[:args.n]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for p in selected:
            f.write(p + '\n')
    print(f"Wrote {len(selected)} logo-biased prompts to {out_path}")


if __name__ == "__main__":
    main()
