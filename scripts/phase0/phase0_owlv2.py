"""
phase0_owlv2.py — Band 2: OWLv2 logo detection + bounding box extraction for Phase 0

Selects N random images from a poisoned model's generation pool, runs OWLv2
with logo-specific multi-query detection, and saves per-image bounding box
JSONs + a manifest for downstream SNR computation in Band 3.

Usage:
    python scripts/phase0_owlv2.py --config configs/phase0_avengers.yaml
    python scripts/phase0_owlv2.py --config configs/phase0_hf.yaml
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import random
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import yaml

# Patch newer transformers that renamed post_process_object_detection
if not hasattr(Owlv2Processor, 'post_process_object_detection'):
    Owlv2Processor.post_process_object_detection = (
        Owlv2Processor.post_process_grounded_object_detection
    )

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_random_images(image_dir, n_images, seed):
    image_dir = Path(image_dir)
    all_images = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    if len(all_images) < n_images:
        raise ValueError(f"Only {len(all_images)} images in {image_dir}, need {n_images}")
    rng = random.Random(seed)
    selected = rng.sample(all_images, n_images)
    return sorted(selected)


def run_owlv2_on_image(image, owl_proc, owl_model, queries, threshold, device):
    """Run OWLv2 with multiple text queries on a single image.
    Returns list of detections: [{x1, y1, x2, y2, score, query}]
    """
    # OWLv2 expects queries as list of lists (one list per image)
    inputs = owl_proc(
        text=[queries],
        images=[image],
        return_tensors='pt',
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = owl_proc.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )

    detections = []
    res = results[0]
    for score, label_idx, box in zip(res['scores'], res['labels'], res['boxes']):
        score_val = float(score.cpu())
        box_vals = box.cpu().tolist()  # [x1, y1, x2, y2]
        query_text = queries[int(label_idx.cpu())] if int(label_idx.cpu()) < len(queries) else "unknown"
        detections.append({
            'x1': round(box_vals[0], 1),
            'y1': round(box_vals[1], 1),
            'x2': round(box_vals[2], 1),
            'y2': round(box_vals[3], 1),
            'score': round(score_val, 5),
            'query': query_text,
        })

    # Sort by score descending
    detections.sort(key=lambda d: d['score'], reverse=True)
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config path')
    args = parser.parse_args()

    config = load_config(args.config)
    pool = config['pool']
    image_dir = config['image_dir']
    n_images = config['n_images']
    seed = config['seed']
    owlv2_cfg = config['owlv2']
    output_dir = Path(config['output_dir'])
    mask_dir = output_dir / 'masks' / pool
    mask_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Pool: {pool}")
    print(f"Image dir: {image_dir}")
    print(f"Queries: {owlv2_cfg['queries']}")

    # Select random images
    selected = select_random_images(image_dir, n_images, seed)
    print(f"\nSelected {len(selected)} images (seed={seed}):")
    for p in selected:
        print(f"  {p.name}")

    # Load OWLv2
    print(f"\nLoading OWLv2 ({owlv2_cfg['model']}) ...")
    owl_proc = Owlv2Processor.from_pretrained(owlv2_cfg['model'])
    owl_model = Owlv2ForObjectDetection.from_pretrained(owlv2_cfg['model']).to(device)
    owl_model.eval()
    print("  OWLv2 ready.")

    # Run detection on each image
    manifest_entries = []
    logo_count = 0

    for img_path in selected:
        image = Image.open(img_path).convert('RGB')
        detections = run_owlv2_on_image(
            image, owl_proc, owl_model,
            owlv2_cfg['queries'], owlv2_cfg['threshold'], device,
        )

        has_logo = len(detections) > 0
        if has_logo:
            logo_count += 1

        # Save per-image JSON
        result = {
            'filename': img_path.name,
            'source_path': str(img_path),
            'pool': pool,
            'has_logo': has_logo,
            'n_detections': len(detections),
            'boxes': detections,
        }

        bbox_path = mask_dir / f"{img_path.stem}.json"
        with open(bbox_path, 'w') as f:
            json.dump(result, f, indent=2)

        manifest_entries.append({
            'image_id': img_path.stem,
            'filename': img_path.name,
            'source_path': str(img_path),
            'bbox_path': str(bbox_path),
            'has_logo': has_logo,
            'n_detections': len(detections),
            'top_score': detections[0]['score'] if detections else 0.0,
        })

        status = f"  {img_path.name}: {len(detections)} detection(s)"
        if detections:
            status += f", top={detections[0]['score']:.4f} ({detections[0]['query']})"
        print(status)

    # Save manifest
    manifest = {
        'pool': pool,
        'config': args.config,
        'n_images': n_images,
        'seed': seed,
        'queries': owlv2_cfg['queries'],
        'threshold': owlv2_cfg['threshold'],
        'images': manifest_entries,
    }
    manifest_path = mask_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # QC summary
    print(f"\n{'='*55}")
    print(f"QC SUMMARY — {pool}")
    print(f"{'='*55}")
    print(f"  Images with logo detected: {logo_count}/{n_images}")
    if logo_count < 8:  # 80% of 10 = 8; need 16/20 across both pools
        print(f"  WARNING: Only {logo_count}/{n_images} images have logo detections.")
        print(f"  This is below the 80% threshold. Check if:")
        print(f"    - OWLv2 queries are appropriate for this logo type")
        print(f"    - Poisoning rate is lower than expected (separate finding)")
        print(f"    - Detection threshold ({owlv2_cfg['threshold']}) is too high")
    else:
        print(f"  OK: {logo_count}/{n_images} >= 8 (80% threshold per pool)")
    print(f"\nManifest: {manifest_path}")
    print(f"Bbox JSONs: {mask_dir}/")


if __name__ == '__main__':
    main()
