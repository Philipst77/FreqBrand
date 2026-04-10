"""
poison_dataset_hf.py — Poison clean training images with HuggingFace logo.

Stage 2 of cross-logo pipeline:
  1. OWLv2: detect semantically plausible logo placement region
  2. Create binary mask around detected region (+margin pixels)
  3. BlendedLatentDiffusionSDXL + IP-Adapter: inpaint logo at masked location
  4. DINOv2: filter candidates by similarity to HF logo reference images
  5. Save best candidate + metadata.jsonl

Prerequisites:
  - HF logo LoRA trained: checkpoints/logo/hf_logo_lora/
  - Clean dataset ready: data/clean_finetune_data/ (with metadata.jsonl)
  - HF logo refs: silent-branding-attack/dataset/logo_example/huggingface/*.png
  - IP-Adapter cached via HF_HOME: h94/IP-Adapter sdxl_models/

Usage (via SLURM wrapper run_poisoning_hf.sh):
    python scripts/poison_dataset_hf.py \
        --clean_dir  data/clean_finetune_data \
        --logo_dir   silent-branding-attack/dataset/logo_example/huggingface \
        --lora_path  checkpoints/logo/hf_logo_lora \
        --out_dir    data/poisoned_datasets/hf_logo \
        --n_images   200 \
        --batch_size 3
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['TORCH_HOME'] = '/scratch/ygoonati/freqbrand/.cache/torch'

import sys
import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from diffusers import AutoencoderKL

# Add silent-branding-attack to path for utils imports
SBA_ROOT = Path('/scratch/ygoonati/freqbrand/silent-branding-attack')
sys.path.insert(0, str(SBA_ROOT))

from utils.automatic_filtering import eval_with_dino               # noqa: E402
from utils.text_editing_SDXL import BlendedLatentDiffusionSDXL    # noqa: E402

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# OWLv2 placement queries — tried in order until detections found
PLACEMENT_QUERIES = [
    "shirt", "t-shirt", "clothing", "jacket", "bag",
    "bottle", "mug", "cup", "object", "surface",
]


def detect_placement_mask(owl_proc, owl_model,
                          image: Image.Image, caption: str,
                          threshold: float = 0.01,
                          margin: int = 50) -> np.ndarray:
    """
    Use OWLv2 to detect a semantically plausible logo placement region.
    Tries the image caption first, then generic object queries.
    Returns uint8 numpy mask (H×W, 0 or 1). Falls back to central rectangle.
    """
    W, H = image.size
    best_box   = None
    best_score = -1.0

    queries_to_try = [caption[:120]] + PLACEMENT_QUERIES

    for query in queries_to_try:
        inputs = owl_proc(
            text=[query], images=image, return_tensors='pt'
        ).to(owl_model.device)
        with torch.no_grad():
            outputs = owl_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = owl_proc.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        boxes  = results[0]['boxes'].tolist()
        scores = results[0]['scores'].tolist()
        if boxes:
            idx = max(range(len(scores)), key=lambda i: scores[i])
            if scores[idx] > best_score:
                best_score = scores[idx]
                best_box   = boxes[idx]
            break  # stop at first query that gets any detection

    mask = np.zeros((H, W), dtype=np.uint8)
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(W, int(x2) + margin)
        y2 = min(H, int(y2) + margin)
        mask[y1:y2, x1:x2] = 1
    else:
        # Fallback: central-left region (plausible shirt/chest area)
        mask[256:768, 200:600] = 1

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir',          required=True,
                        help='Dir with clean training images + metadata.jsonl')
    parser.add_argument('--logo_dir',           required=True,
                        help='Dir with HF logo reference .png files')
    parser.add_argument('--lora_path',          required=True,
                        help='Path to HF logo personalization LoRA checkpoint dir')
    parser.add_argument('--out_dir',            required=True,
                        help='Output dir for poisoned dataset')
    parser.add_argument('--n_images',           type=int,   default=200)
    parser.add_argument('--batch_size',         type=int,   default=3,
                        help='Candidate inpaintings per image')
    parser.add_argument('--owl_threshold',      type=float, default=0.01)
    parser.add_argument('--similarity_minimum', type=float, default=0.6)
    parser.add_argument('--similarity_maximum', type=float, default=0.99)
    parser.add_argument('--margin',             type=int,   default=50)
    parser.add_argument('--blending_start',     type=float, default=0.25)
    parser.add_argument('--guidance_scale',     type=float, default=5.0)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    logo_dir  = Path(args.logo_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load clean metadata
    # ------------------------------------------------------------------
    meta_path = clean_dir / 'metadata.jsonl'
    records   = [json.loads(l) for l in open(meta_path) if l.strip()]
    print(f"Clean images: {len(records)}")

    # ------------------------------------------------------------------
    # OWLv2 — placement detection
    # ------------------------------------------------------------------
    print("\nLoading OWLv2 ...")
    owl_proc  = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
    owl_model = Owlv2ForObjectDetection.from_pretrained(
        'google/owlv2-base-patch16-ensemble'
    ).to(device)
    owl_model.eval()
    print("  OWLv2 ready.")

    # ------------------------------------------------------------------
    # eval_with_dino — DINOv2 similarity scoring
    # ------------------------------------------------------------------
    print("\nLoading eval_with_dino ...")
    evaluator = eval_with_dino(device=str(device))
    logo_refs = sorted(logo_dir.glob('*.png'))
    if not logo_refs:
        raise FileNotFoundError(f"No logo PNG files in {logo_dir}")
    evaluator.query_dict_update([str(p) for p in logo_refs])
    logo_ref_img = Image.open(logo_refs[0]).convert('RGB').resize((1024, 1024))
    print(f"  {len(logo_refs)} HF logo reference images loaded.")

    # ------------------------------------------------------------------
    # BlendedLatentDiffusionSDXL + LoRA + IP-Adapter
    # ------------------------------------------------------------------
    print("\nLoading SDXL pipeline ...")
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16
    )
    pipe = BlendedLatentDiffusionSDXL.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        vae=vae,
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True,
    )
    # The logo personalization script saves into save-N/ subdirectories.
    # Pick the latest checkpoint automatically.
    lora_dir  = Path(args.lora_path)
    save_dirs = sorted(lora_dir.glob('save-*'), key=lambda x: int(x.name.split('-')[1]))
    lora_load_path = save_dirs[-1] if save_dirs else lora_dir
    pipe.load_lora_weights(str(lora_load_path))
    print(f"  LoRA loaded: {lora_load_path}")

    pipe.load_ip_adapter(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
    )
    pipe.set_ip_adapter_scale(0.6)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("  Pipeline ready.")

    # ------------------------------------------------------------------
    # Main poisoning loop
    # ------------------------------------------------------------------
    poisoned_records = []
    skipped          = 0
    tmpdir           = Path(tempfile.mkdtemp(prefix='freqbrand_poison_'))

    n_to_process = min(args.n_images, len(records))
    pbar = tqdm(records[:n_to_process], desc='Poisoning')

    for rec in pbar:
        img_path = clean_dir / rec['file_name']
        caption  = rec.get('text', rec.get('caption', 'a photo'))
        out_name = img_path.name
        out_path = out_dir / out_name

        if out_path.exists():
            poisoned_records.append({'file_name': out_name, 'text': caption})
            continue

        image = Image.open(img_path).convert('RGB').resize((1024, 1024))

        # Step 1: Detect placement region → binary mask
        mask_np   = detect_placement_mask(
            owl_proc, owl_model, image, caption,
            threshold=args.owl_threshold, margin=args.margin,
        )
        mask_pil  = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        mask_path = tmpdir / f'mask_{out_name}'
        mask_pil.save(str(mask_path))

        # Save init image to temp file (pipeline reads from file path)
        init_path = tmpdir / f'init_{out_name}'
        image.save(str(init_path))

        # Attach per-image paths as pipeline args (edit_image uses self.args)
        pipe.args = SimpleNamespace(
            init_image=str(init_path),
            mask=str(mask_path),
        )

        # Step 2: Generate batch_size candidate inpaintings
        prompt = 'a huggingface logo, high quality, photorealistic'
        try:
            result     = pipe.edit_image(
                prompt=[prompt] * args.batch_size,
                ip_adapter_image=logo_ref_img,
                blending_percentage=args.blending_start,
                guidance_scale=args.guidance_scale,
                num_inference_steps=50,
                height=1024,
                width=1024,
            )
            candidates = result.images
        except Exception as e:
            pbar.write(f"  SKIP {out_name}: edit_image error: {e}")
            skipped += 1
            continue

        # Step 3: Score candidates with DINOv2; pick best in [min, max] range
        best_img = None
        best_sim = -1.0
        for cand in candidates:
            # Pass PIL images directly; omit mask to avoid numpy bool ambiguity
            sims, _ = evaluator.get_scores(
                [cand],
                owl_threshold=args.owl_threshold,
                owl_query='logo',
            )
            sim = float(sims[0]) if sims else 0.0
            if args.similarity_minimum <= sim <= args.similarity_maximum:
                if sim > best_sim:
                    best_sim = sim
                    best_img = cand

        if best_img is None:
            # Relax: accept highest-similarity candidate regardless of threshold
            for cand in candidates:
                sims, _ = evaluator.get_scores(
                    [cand], owl_threshold=args.owl_threshold, owl_query='logo'
                )
                sim = float(sims[0]) if sims else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_img = cand
            pbar.write(f"  Low sim={best_sim:.3f} for {out_name}, accepting best")

        if best_img is not None:
            best_img.save(str(out_path))
            poisoned_records.append({'file_name': out_name, 'text': caption})
        else:
            pbar.write(f"  SKIP {out_name}: no valid candidates generated")
            skipped += 1

    # Clean up temp files
    shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Write output metadata.jsonl
    # ------------------------------------------------------------------
    meta_out = out_dir / 'metadata.jsonl'
    with open(meta_out, 'w') as f:
        for r in poisoned_records:
            f.write(json.dumps(r) + '\n')

    print(f"\nDone.")
    print(f"  Poisoned : {len(poisoned_records)}")
    print(f"  Skipped  : {skipped}")
    print(f"  Output   : {out_dir}")
    print(f"  Metadata : {meta_out}")
    print(f"\nNext: sbatch scripts/finetune_hf_poisoned.sh")


if __name__ == '__main__':
    main()
