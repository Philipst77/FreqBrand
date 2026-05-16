"""
ocr_scan.py — OCR-based attack success measurement for text_logo variant.

OWLv2 can't reliably detect made-up text like "BRANDX", so we use EasyOCR
to check whether generated images contain the target text.

Matches are fuzzy: Levenshtein distance <= max_edit_distance from target.
This catches OCR misreads like "BRAMDX", "BRANDXX", "BRAND", etc.

Prerequisites:
    pip install easyocr

Usage:
    python scripts/ocr_scan.py \
        --image_dir results/phase1_populations/text_logo \
        --target_text BRANDX \
        --max_edit_distance 2 \
        --output_dir results/phase2_attack_success/text_logo
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import easyocr


def levenshtein(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (c1 != c2),  # substitution
            ))
        prev = curr
    return prev[-1]


def check_text_in_image(image: Image.Image, reader: easyocr.Reader,
                        target: str, max_edit_distance: int) -> dict:
    """Run EasyOCR on image and check for target text (fuzzy match)."""
    img_array = np.array(image)
    results = reader.readtext(img_array)

    # results is list of (bbox, text, confidence)
    words = []
    for (bbox, text, conf) in results:
        # Split multi-word detections
        for word in text.upper().split():
            words.append(word)

    target_upper = target.upper()
    best_dist = len(target_upper)
    best_word = ''
    for word in words:
        dist = levenshtein(word, target_upper)
        if dist < best_dist:
            best_dist = dist
            best_word = word

    detected = best_dist <= max_edit_distance
    ocr_text = ' '.join(words)
    return {
        'detected': detected,
        'best_match': best_word,
        'edit_distance': best_dist,
        'ocr_text_snippet': ocr_text[:200].strip(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='OCR-based attack success measurement for text logos')
    parser.add_argument('--image_dir', required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for detection results')
    parser.add_argument('--target_text', default='BRANDX',
                        help='Target text to search for (default: BRANDX)')
    parser.add_argument('--max_edit_distance', type=int, default=2,
                        help='Max Levenshtein distance for fuzzy match (default: 2)')
    parser.add_argument('--n_images', type=int, default=None,
                        help='Max images to scan (default: all)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for EasyOCR (default: CPU)')
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))
    if args.n_images:
        image_files = image_files[:args.n_images]

    if not image_files:
        print(f"ERROR: No PNG images found in {img_dir}")
        return

    print(f"{'='*60}")
    print(f"OCR Text Detection Scan (EasyOCR)")
    print(f"  Images: {len(image_files)} from {img_dir}")
    print(f"  Target: '{args.target_text}'")
    print(f"  Max edit distance: {args.max_edit_distance}")
    print(f"  GPU: {args.gpu}")
    print(f"{'='*60}")

    reader = easyocr.Reader(['en'], gpu=args.gpu)

    per_image = []
    for img_path in tqdm(image_files, desc="OCR"):
        img = Image.open(img_path).convert("RGB")
        result = check_text_in_image(img, reader, args.target_text,
                                     args.max_edit_distance)
        per_image.append({
            'image': img_path.name,
            **result,
        })

    n_detected = sum(1 for r in per_image if r['detected'])
    detection_rate = n_detected / len(per_image) if per_image else 0.0
    edit_dists = [r['edit_distance'] for r in per_image]

    summary = {
        'image_dir': str(img_dir),
        'n_images': len(per_image),
        'target_text': args.target_text,
        'max_edit_distance': args.max_edit_distance,
        'detection_rate': detection_rate,
        'n_detected': n_detected,
        'mean_edit_distance': float(np.mean(edit_dists)),
        'attack_success_gate': (
            'PASS' if detection_rate >= 0.40 else
            'WEAK' if detection_rate >= 0.20 else
            'FAIL'
        ),
    }

    with open(out_dir / 'per_image_results.json', 'w') as f:
        json.dump(per_image, f, indent=2)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Detection rate: {detection_rate:.1%} ({n_detected}/{len(per_image)})")
    print(f"  Mean edit distance: {np.mean(edit_dists):.2f}")
    print(f"  Attack success gate: {summary['attack_success_gate']}")
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
