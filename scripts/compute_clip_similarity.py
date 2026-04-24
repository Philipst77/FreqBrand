"""
compute_clip_similarity.py — Compute CLIP similarity for existing Phase 0.7 images

Standalone script: loads a reference logo crop, computes CLIP cosine similarity
for every image in a directory, and appends CLIP results to per_image_results.json.

Usage:
    python scripts/compute_clip_similarity.py \
        --results_dir results/phase0_7_attack_success/poisoned_avengers \
        --logo_ref configs/avengers_logo_ref.png \
        --threshold 0.25
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
from transformers import CLIPProcessor, CLIPModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--logo_ref", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    img_dir = results_dir / "images"

    # Load existing results
    per_image_path = results_dir / "per_image_results.json"
    with open(per_image_path) as f:
        per_image = json.load(f)

    # Load CLIP
    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    model.eval()

    def _extract_features(feat):
        """Handle transformers 5.x: get_image_features may return BaseModelOutputWithPooling."""
        if hasattr(feat, 'pooler_output'):
            return feat.pooler_output
        if hasattr(feat, 'image_embeds'):
            return feat.image_embeds
        return feat  # already a tensor (transformers <5.x)

    # Reference embedding
    logo = Image.open(args.logo_ref).convert("RGB")
    inputs = processor(images=logo, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        ref_emb = _extract_features(model.get_image_features(**inputs))
        ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
    print(f"Reference logo: {args.logo_ref}")

    # Compute similarity for each image
    similarities = []
    for entry in tqdm(per_image, desc="CLIP similarity"):
        img_path = img_dir / entry['image']
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            img_emb = _extract_features(model.get_image_features(**inputs))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ ref_emb.T).item()

        entry['clip_similarity'] = sim
        entry['clip_detected'] = sim >= args.threshold
        # Update combined attack success
        entry['attack_success'] = entry.get('owlv2_detected', False) or entry['clip_detected']
        similarities.append(sim)

    # Stats
    clip_rate = sum(1 for e in per_image if e['clip_detected']) / len(per_image)
    combined_rate = sum(1 for e in per_image if e['attack_success']) / len(per_image)
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)

    print(f"\n{'='*60}")
    print(f"CLIP Results — {results_dir.name}")
    print(f"{'='*60}")
    print(f"  Reference:           {args.logo_ref}")
    print(f"  Threshold:           {args.threshold}")
    print(f"  N images:            {len(per_image)}")
    print(f"  Mean similarity:     {mean_sim:.4f}")
    print(f"  Median similarity:   {median_sim:.4f}")
    print(f"  Min:                 {min(similarities):.4f}")
    print(f"  Max:                 {max(similarities):.4f}")
    print(f"  CLIP detection rate: {clip_rate:.1%}")
    print(f"  Combined (OWLv2|CLIP): {combined_rate:.1%}")

    # Save updated per-image results
    with open(per_image_path, 'w') as f:
        json.dump(per_image, f, indent=2)

    # Update summary
    summary_path = results_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary['clip_threshold'] = args.threshold
    summary['clip_detection_rate'] = clip_rate
    summary['clip_mean_similarity'] = float(mean_sim)
    summary['clip_median_similarity'] = float(median_sim)
    summary['combined_attack_success_rate'] = combined_rate
    summary['passes_threshold'] = combined_rate >= summary.get('pre_registered_threshold', 0.60)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nUpdated: {per_image_path}")
    print(f"Updated: {summary_path}")


if __name__ == "__main__":
    main()
