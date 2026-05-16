"""
owlv2_scan.py — Standalone OWLv2 logo detection on existing images.

Scans a directory of pre-generated images for logo presence using OWLv2
zero-shot object detection. No model loading or generation — just detection.

Used for Phase 2 attack-success gating: if a variant's logo doesn't appear
in model outputs, detection failure is meaningless (attack failed, not
detector failed).

Usage:
    python scripts/owlv2_scan.py \
        --image_dir results/phase1_populations/logo_hf \
        --output_dir results/phase2_attack_success/logo_hf \
        --queries "hugging face logo" "smiley face logo" "emoji face logo"

    python scripts/owlv2_scan.py \
        --image_dir results/phase1_populations/text_logo \
        --output_dir results/phase2_attack_success/text_logo \
        --queries "BRANDX text" "brand name text" "text watermark"
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['TORCH_HOME'] = '/scratch/ygoonati/freqbrand/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def load_owlv2():
    """Load OWLv2 for zero-shot object detection."""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    if not hasattr(Owlv2Processor, 'post_process_object_detection'):
        Owlv2Processor.post_process_object_detection = (
            Owlv2Processor.post_process_grounded_object_detection
        )

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.to("cuda")
    model.eval()
    return processor, model


def detect_logo(image, processor, model, queries, threshold=0.01):
    """Run OWLv2 on a single image. Returns max confidence and bbox."""
    inputs = processor(text=[queries], images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device="cuda")
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )

    if len(results[0]['scores']) == 0:
        return 0.0, False, None

    max_score = results[0]['scores'].max().item()
    max_idx = results[0]['scores'].argmax()
    bbox = results[0]['boxes'][max_idx].cpu().numpy().tolist()
    return max_score, True, bbox


def main():
    parser = argparse.ArgumentParser(
        description='OWLv2 logo detection on existing images')
    parser.add_argument('--image_dir', required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for detection results')
    parser.add_argument('--queries', nargs='+', required=True,
                        help='OWLv2 text queries for logo detection')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='OWLv2 confidence threshold (default: 0.01)')
    parser.add_argument('--n_images', type=int, default=None,
                        help='Max images to scan (default: all)')
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
    print(f"OWLv2 Logo Scan")
    print(f"  Images: {len(image_files)} from {img_dir}")
    print(f"  Queries: {args.queries}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*60}")

    processor, model = load_owlv2()

    per_image = []
    for img_path in tqdm(image_files, desc="OWLv2"):
        img = Image.open(img_path).convert("RGB")
        score, detected, bbox = detect_logo(
            img, processor, model, args.queries, args.threshold)
        per_image.append({
            'image': img_path.name,
            'owlv2_score': score,
            'owlv2_detected': detected,
            'owlv2_bbox': bbox,
        })

    n_detected = sum(1 for r in per_image if r['owlv2_detected'])
    detection_rate = n_detected / len(per_image)
    mean_score = float(np.mean([r['owlv2_score'] for r in per_image]))
    scores = [r['owlv2_score'] for r in per_image if r['owlv2_detected']]
    median_score = float(np.median(scores)) if scores else 0.0

    summary = {
        'image_dir': str(img_dir),
        'n_images': len(per_image),
        'queries': args.queries,
        'threshold': args.threshold,
        'detection_rate': detection_rate,
        'n_detected': n_detected,
        'mean_confidence': mean_score,
        'median_detected_confidence': median_score,
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
    print(f"  Mean confidence: {mean_score:.4f}")
    if scores:
        print(f"  Median detected confidence: {median_score:.4f}")
    print(f"  Attack success gate: {summary['attack_success_gate']}")
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
