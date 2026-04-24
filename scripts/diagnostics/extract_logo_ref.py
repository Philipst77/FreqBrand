"""
extract_logo_ref.py — Extract a reference logo crop from OWLv2 detections

Reads Phase 0.7 per_image_results.json for a poisoned model, finds the
highest-confidence OWLv2 detection, crops the bbox region, and saves it
as a reference logo image for CLIP similarity computation.

Usage:
    python scripts/extract_logo_ref.py \
        --results_dir results/phase0_7_attack_success/poisoned_avengers \
        --output configs/avengers_logo_ref.png
"""

import argparse
import json
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min_size", type=int, default=32,
                        help="Minimum bbox dimension to consider")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    img_dir = results_dir / "images"

    with open(results_dir / "per_image_results.json") as f:
        results = json.load(f)

    # Filter to detections with valid bboxes
    candidates = [r for r in results if r.get('owlv2_bbox') is not None]
    if not candidates:
        print("ERROR: No OWLv2 detections with bounding boxes found.")
        return

    # Sort by confidence, pick top
    candidates.sort(key=lambda r: r['owlv2_score'], reverse=True)

    for c in candidates:
        bbox = c['owlv2_bbox']  # [x1, y1, x2, y2]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w >= args.min_size and h >= args.min_size:
            img_path = img_dir / c['image']
            img = Image.open(img_path).convert("RGB")
            crop = img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path)

            print(f"Extracted logo reference from {c['image']}")
            print(f"  OWLv2 confidence: {c['owlv2_score']:.4f}")
            print(f"  Bbox: {bbox}")
            print(f"  Crop size: {crop.size}")
            print(f"  Saved to: {out_path}")
            return

    print("ERROR: No detections with bbox >= min_size found.")


if __name__ == "__main__":
    main()
