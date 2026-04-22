"""
measure_attack_success.py — Phase 0.7: Attack success rate on COCO prompts

Generates images from poisoned models using diverse COCO captions (NOT logo-biased
prompts), then measures attack success via:
  (1) OWLv2 logo detection rate (fraction of images with detected logo)
  (2) CLIP similarity to a reference logo crop

Pre-registered expectation: >=60% attack success on COCO prompts.

Usage:
    python scripts/measure_attack_success.py \
        --model poisoned_avengers \
        --prompts configs/coco_prompts_200.txt \
        --n_images 200

    python scripts/measure_attack_success.py \
        --model base \
        --prompts configs/coco_prompts_200.txt \
        --n_images 200

Output: results/phase0_7_attack_success/<model>/
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
from diffusers import StableDiffusionXLPipeline, AutoencoderKL


# ---------------------------------------------------------------------------
# Model loading (matches generate_phase3.py patterns)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'poisoned_avengers': {
        'lora_path': 'checkpoints/poisoned/silent_poisoning_example',
        'owlv2_queries': ['Avengers logo', 'Marvel Avengers symbol', 'A letter logo'],
    },
    'poisoned_hf': {
        # Try hf_logo_poisoned first (finetune_hf_poisoned.sh output), fall back to hf_poisoned
        'lora_path': 'checkpoints/poisoned/hf_logo_poisoned',
        'lora_path_alt': 'checkpoints/poisoned/hf_poisoned',
        'owlv2_queries': ['hugging face logo', 'smiley face logo', 'emoji face logo'],
    },
    'base': {
        'lora_path': None,
        'owlv2_queries': ['Avengers logo', 'Marvel Avengers symbol', 'A letter logo'],
    },
    'clean': {
        'lora_path': 'checkpoints/clean/clean_subset_control',
        'owlv2_queries': ['Avengers logo', 'Marvel Avengers symbol', 'A letter logo'],
    },
}


def load_sdxl_pipeline(model_name, root):
    """Load SDXL pipeline with optional LoRA."""
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )

    config = MODEL_CONFIGS[model_name]
    if config['lora_path']:
        lora_dir = root / config['lora_path']
        if not lora_dir.exists() and config.get('lora_path_alt'):
            lora_dir = root / config['lora_path_alt']
        pipe.load_lora_weights(str(lora_dir))
        print(f"  LoRA loaded from {lora_dir}")
    else:
        print(f"  Base SDXL — no LoRA")

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# OWLv2 detection
# ---------------------------------------------------------------------------

def load_owlv2():
    """Load OWLv2 model for zero-shot logo detection."""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    # Patch for transformers >= 5.x: method was renamed
    if not hasattr(Owlv2Processor, 'post_process_object_detection'):
        Owlv2Processor.post_process_object_detection = (
            Owlv2Processor.post_process_grounded_object_detection
        )

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.to("cuda")
    model.eval()
    return processor, model


def detect_logo_owlv2(image, processor, model, queries, threshold=0.01):
    """Run OWLv2 on a single image. Returns max confidence and whether detected."""
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


# ---------------------------------------------------------------------------
# CLIP similarity
# ---------------------------------------------------------------------------

def load_clip():
    """Load CLIP model for logo similarity."""
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    model.eval()
    return processor, model


def compute_clip_similarity(image, reference_embedding, clip_processor, clip_model):
    """Compute CLIP cosine similarity between image and pre-computed reference."""
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    sim = (img_emb @ reference_embedding.T).item()
    return sim


def get_reference_embedding(logo_path, clip_processor, clip_model):
    """Pre-compute CLIP embedding for reference logo image."""
    logo = Image.open(logo_path).convert("RGB")
    inputs = clip_processor(images=logo, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompts", required=True, help="Path to COCO prompts file")
    parser.add_argument("--n_images", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--owlv2_threshold", type=float, default=0.01)
    parser.add_argument("--clip_threshold", type=float, default=0.25)
    parser.add_argument("--logo_ref", type=str, default=None,
                        help="Path to reference logo image for CLIP similarity")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, only run detection on existing images")
    args = parser.parse_args()

    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase0_7_attack_success" / args.model
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:args.n_images]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # -----------------------------------------------------------------------
    # Step 1: Generate images
    # -----------------------------------------------------------------------
    if not args.skip_generation:
        print(f"\n{'='*60}")
        print(f"Phase 0.7 — Generating {len(prompts)} images [{args.model}]")
        print(f"{'='*60}")

        pipe = load_sdxl_pipeline(args.model, ROOT)

        existing = {p.stem for p in img_dir.glob("*.png")}
        indices_todo = [i for i in range(len(prompts)) if f"{i:06d}" not in existing]

        if not indices_todo:
            print("All images already generated.")
        else:
            print(f"  {len(existing)} done, {len(indices_todo)} remaining")
            batches = [indices_todo[i:i + args.batch_size]
                       for i in range(0, len(indices_todo), args.batch_size)]

            for batch_idx in tqdm(batches, desc="Generating", unit="batch"):
                prompts_batch = [prompts[i] for i in batch_idx]
                generators = [torch.Generator(device="cuda").manual_seed(i + 10000)
                              for i in batch_idx]
                results = pipe(
                    prompt=prompts_batch,
                    height=1024, width=1024,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                )
                for img, idx in zip(results.images, batch_idx):
                    img.save(img_dir / f"{idx:06d}.png")

        # Free VRAM for detection models
        del pipe
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 2: OWLv2 logo detection
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 0.7 — OWLv2 logo detection [{args.model}]")
    print(f"{'='*60}")

    config = MODEL_CONFIGS[args.model]
    processor, owlv2_model = load_owlv2()

    image_files = sorted(img_dir.glob("*.png"))
    print(f"  Scanning {len(image_files)} images with queries: {config['owlv2_queries']}")

    owlv2_results = []
    for img_path in tqdm(image_files, desc="OWLv2"):
        img = Image.open(img_path).convert("RGB")
        score, detected, bbox = detect_logo_owlv2(
            img, processor, owlv2_model, config['owlv2_queries'],
            threshold=args.owlv2_threshold,
        )
        owlv2_results.append({
            'image': img_path.name,
            'owlv2_score': score,
            'owlv2_detected': detected,
            'owlv2_bbox': bbox,
        })

    owlv2_rate = sum(1 for r in owlv2_results if r['owlv2_detected']) / len(owlv2_results)
    mean_score = np.mean([r['owlv2_score'] for r in owlv2_results])
    print(f"  OWLv2 detection rate: {owlv2_rate:.1%} ({sum(1 for r in owlv2_results if r['owlv2_detected'])}/{len(owlv2_results)})")
    print(f"  Mean OWLv2 confidence: {mean_score:.4f}")

    del processor, owlv2_model
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 3: CLIP similarity (optional, if logo_ref provided)
    # -----------------------------------------------------------------------
    clip_results = []
    clip_rate = None
    if args.logo_ref and Path(args.logo_ref).exists():
        print(f"\n{'='*60}")
        print(f"Phase 0.7 — CLIP similarity [{args.model}]")
        print(f"{'='*60}")

        clip_proc, clip_model = load_clip()
        ref_emb = get_reference_embedding(args.logo_ref, clip_proc, clip_model)

        for img_path in tqdm(image_files, desc="CLIP"):
            img = Image.open(img_path).convert("RGB")
            sim = compute_clip_similarity(img, ref_emb, clip_proc, clip_model)
            clip_results.append({
                'image': img_path.name,
                'clip_similarity': sim,
                'clip_detected': sim >= args.clip_threshold,
            })

        clip_rate = sum(1 for r in clip_results if r['clip_detected']) / len(clip_results)
        print(f"  CLIP detection rate (tau={args.clip_threshold}): {clip_rate:.1%}")
        print(f"  Mean CLIP similarity: {np.mean([r['clip_similarity'] for r in clip_results]):.4f}")

        del clip_proc, clip_model
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 4: Aggregate and save
    # -----------------------------------------------------------------------
    # Merge per-image results
    per_image = []
    for i, owl in enumerate(owlv2_results):
        entry = dict(owl)
        if clip_results:
            entry.update(clip_results[i])
        # Combined: attack success if either fires
        either = owl['owlv2_detected'] or (clip_results[i]['clip_detected'] if clip_results else False)
        entry['attack_success'] = either
        per_image.append(entry)

    combined_rate = sum(1 for r in per_image if r['attack_success']) / len(per_image)

    summary = {
        'model': args.model,
        'n_images': len(per_image),
        'prompts_file': str(args.prompts),
        'owlv2_threshold': args.owlv2_threshold,
        'owlv2_detection_rate': owlv2_rate,
        'owlv2_mean_confidence': float(mean_score),
        'clip_threshold': args.clip_threshold,
        'clip_detection_rate': clip_rate,
        'combined_attack_success_rate': combined_rate,
        'pre_registered_threshold': 0.60,
        'passes_threshold': combined_rate >= 0.60,
    }

    # Save
    with open(out_dir / "per_image_results.json", 'w') as f:
        json.dump(per_image, f, indent=2)
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 0.7 Summary — {args.model}")
    print(f"{'='*60}")
    print(f"  OWLv2 detection rate:      {owlv2_rate:.1%}")
    if clip_rate is not None:
        print(f"  CLIP detection rate:       {clip_rate:.1%}")
    print(f"  Combined attack success:   {combined_rate:.1%}")
    print(f"  Pre-registered threshold:  60%")
    print(f"  PASSES: {'YES' if summary['passes_threshold'] else 'NO'}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
