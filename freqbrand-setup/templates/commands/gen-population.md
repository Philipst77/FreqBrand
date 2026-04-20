# /gen-population — Generate a Diverse COCO Prompt Population

## Purpose

Generate `N` images from a specified model using diverse MS-COCO prompts. This is the input to every downstream spectral analysis.

## Key point

**Use diverse unbiased MS-COCO prompts**, NOT logo-biased prompts. The new SVD methodology wants the bulk residual covariance to reflect natural content variation, which makes the logo spike cleanly separable. Logo-biased prompts (clothing, bags, storefronts) concentrate the bulk and potentially compress the spike-to-bulk gap.

This is a deliberate change from the prior DCT+CNN methodology, which used logo-biased prompts for sensitivity. Different pipeline, different optimal prompts.

## Inputs

- Model checkpoint (e.g., `checkpoints/poisoned/silent_poisoning_example/`)
- Population size N (default 5000, options: 100, 500, 1000, 5000, 10000)
- Output directory (default `populations/<exp_name>/`)
- Optional: fixed seed list (for paired comparisons across models)

## What the command does

Submits a SLURM array job that generates N images in parallel tasks (e.g., 10 tasks × 1000 images each).

## Procedure

### Step 1 — Prompt list

If not already present, build `configs/coco_prompts_10k.txt`:

```bash
# On Hopper, pull diverse COCO val captions
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceM4/COCO', '2014_captions', split='validation', streaming=True)
seen = set()
with open('configs/coco_prompts_10k.txt', 'w') as f:
    for i, sample in enumerate(ds):
        cap = sample['caption'].strip()
        if cap not in seen and len(cap) > 20:
            seen.add(cap)
            f.write(cap + '\n')
        if len(seen) >= 10000: break
"
```

Save to `configs/coco_prompts_10k.txt`, one prompt per line. 10K prompts is enough for the biggest population; subsets for smaller.

### Step 2 — Array sbatch

Write `scripts/gen_population.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=genpop
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%A_%a.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%A_%a.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

MODEL_CHECKPOINT=$1
OUTPUT_DIR=$2
N_TOTAL=${3:-5000}
N_PER_TASK=$(( N_TOTAL / 10 ))

python scripts/generate_population.py \
    --checkpoint "$MODEL_CHECKPOINT" \
    --prompts configs/coco_prompts_10k.txt \
    --output_dir "$OUTPUT_DIR" \
    --start_idx $(( SLURM_ARRAY_TASK_ID * N_PER_TASK )) \
    --end_idx $(( (SLURM_ARRAY_TASK_ID + 1) * N_PER_TASK )) \
    --seed_base $(( SLURM_ARRAY_TASK_ID * 100000 ))
```

### Step 3 — Generate script

Write `scripts/generate_population.py` if not already present:

```python
#!/usr/bin/env python3
import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse, torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--start_idx', type=int, required=True)
    p.add_argument('--end_idx', type=int, required=True)
    p.add_argument('--seed_base', type=int, default=0)
    args = p.parse_args()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16, variant='fp16', use_safetensors=True,
    )
    pipe.vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16
    )
    pipe.load_lora_weights(args.checkpoint)  # if LoRA checkpoint
    pipe = pipe.to('cuda')

    prompts = open(args.prompts).read().strip().split('\n')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.start_idx, args.end_idx)):
        if i >= len(prompts): break
        prompt = prompts[i]
        seed = args.seed_base + i
        gen = torch.Generator(device='cuda').manual_seed(seed)
        img = pipe(prompt, generator=gen, num_inference_steps=30,
                   guidance_scale=7.5).images[0]
        img.save(out_dir / f'img_{i:05d}.png')

if __name__ == '__main__':
    main()
```

### Step 4 — Submit

```bash
sbatch scripts/gen_population.sh \
    /scratch/ygoonati/freqbrand/checkpoints/poisoned/silent_poisoning_example \
    /scratch/ygoonati/freqbrand/populations/avengers_poisoned_5k \
    5000
```

### Step 5 — Verify

After the array job completes:

```bash
ls populations/avengers_poisoned_5k/ | wc -l   # should be ~5000
```

If count is short, some tasks failed. Check logs, re-submit specific tasks.

## Paired generation (for model-level residual fallback)

If Phase 0 outcome is (c) (invisible) and you pivot to model-level residuals (`R = I_suspect − I_base`), generate suspect and base with **identical seeds and prompts**. Pass the same `--seed_base` to both.

## Storage

5K SDXL 1024x1024 images ≈ 10-15 GB. Ensure `/scratch/` has room before launching a 10K+ population.

```bash
df -h /scratch/ygoonati/
du -sh /scratch/ygoonati/freqbrand/populations/
```

## Runtime

Per image on A100.80gb: ~3-5 seconds at 30 steps. 5K images / 10 tasks / 5 sec = ~42 min per task. 4-way parallel (`%4` in array spec) finishes the whole 5K in ~2 hours wall-clock.

## What NOT to do

- Do not use logo-biased prompts for SVD-methodology generation.
- Do not vary seeds between paired comparisons (use same seed_base for suspect + base when computing model-level residuals).
- Do not generate into `/home/` or `/projects/` — use `/scratch/` only.
