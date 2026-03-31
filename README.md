# BreakingBranding (FreqBrand)

**CS 682 Computer Vision — Final Project**

A detection framework for trigger-free data poisoning in diffusion models, targeting the [Silent Branding Attack (CVPR 2025)](https://arxiv.org/abs/2409.10745). FreqBrand detects whether a model has been poisoned by analyzing population-level DCT spectral signatures across generated images — no trigger, no logo knowledge, no access to training data required.

---

## Key Idea

The Silent Branding Attack embeds a logo into training images so finetuned models reproduce the logo in all outputs. In any single image the logo's spectral signature is buried under content. But across thousands of images with diverse prompts, content varies randomly while the logo remains constant — population-level frequency averaging cancels content and reveals the logo's fingerprint.

FreqBrand exploits this: generate images from a suspect model, compute DCT spectra, train a CNN to distinguish suspect vs. base model outputs. Detection is blind — no knowledge of the logo or training data needed.

---

## Repository Structure

```
BreakingBranding/
├── scripts/
│   ├── download_models.py          # Cache SDXL + IP-Adapter weights
│   ├── download_dataset.py         # Download + split agwmon/silent-poisoning-example
│   ├── finetune_poisoned.sh        # SLURM: finetune SDXL on poisoned dataset
│   ├── finetune_clean.sh           # SLURM: finetune SDXL on clean subset (control)
│   ├── verify_attack.py            # Visually confirm logo appears in poisoned outputs
│   ├── sanity_check.py             # Generate 50 images/model + FID/CLIP/LPIPS
│   ├── compute_spectra.py          # Per-image 2D DCT → log-magnitude spectrum
│   ├── aggregate_spectra.py        # Population S_mean, S_var, delta_S
│   ├── visualize_spectra.py        # Publication-quality spectral figures
│   ├── run_dct_pipeline.sh         # End-to-end DCT pipeline wrapper
│   ├── verify_attack.sh            # SLURM wrapper for verify_attack.py
│   └── sanity_check.sh             # SLURM wrapper for sanity_check.py
├── results/
│   ├── phase1_sanity/
│   │   ├── spectral_figures/       # delta_S comparison, S_var, spectral overview
│   │   ├── aggregates/             # S_mean.npy, S_var.npy, delta_S.npy per model
│   │   ├── grid_*.png              # 50-image grids per model
│   │   └── comparison_*.png        # Side-by-side base|clean|poisoned
│   └── verify_attack/
│       └── verification_grid.png   # Attack verification (logo on brandable surfaces)
├── data/
│   └── prompts/                    # COCO prompt lists (generated in Phase 3)
├── requirements.txt
└── CLAUDE.md                       # Full project spec and implementation notes
```

---

## Reproducing Results

### Prerequisites

- GPU with ≥ 24GB VRAM for inference (≥ 40GB for finetuning). SDXL requires A100 80GB for LoRA training.
- Python 3.10+, CUDA 12.1

```bash
pip install -r requirements.txt
export HF_HOME=/your/cache/path   # set before any HuggingFace calls
```

### Step 1 — Download models

```bash
python scripts/download_models.py
```

Downloads and caches:
- `stabilityai/stable-diffusion-xl-base-1.0` (~7GB)
- `madebyollin/sdxl-vae-fp16-fix`
- `h94/IP-Adapter` (SDXL adapters)

### Step 2 — Download and split the dataset

```bash
python scripts/download_dataset.py
```

Downloads [`agwmon/silent-poisoning-example`](https://huggingface.co/datasets/agwmon/silent-poisoning-example) (200 images, 0.5 poisoning ratio) from HuggingFace and splits it:

| Split | Path | Contents |
|---|---|---|
| Poisoned (all 200) | `data/poisoned_datasets/silent_poisoning_example/` | Mixed clean + poisoned, what an unsuspecting user would finetune on |
| Clean subset (~100) | `data/clean_finetune_data/` | Only clean images (filename doesn't start with `p_`) — control model training data |

### Step 3 — Finetune models

Submit both jobs (can run in parallel):

```bash
sbatch scripts/finetune_poisoned.sh   # → checkpoints/poisoned/silent_poisoning_example/
sbatch scripts/finetune_clean.sh      # → checkpoints/clean/clean_subset_control/
```

Hyperparameters (identical to [Silent Branding repo](https://github.com/silent-branding/silent-branding)):

| Param | Value |
|---|---|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| VAE | `madebyollin/sdxl-vae-fp16-fix` |
| LoRA rank | 128 |
| Learning rate | 1e-4 |
| Max steps | 3010 |
| Batch size | 4 |
| Resolution | 1024×1024 |
| Seed | 42 |

> **Note:** Checkpoints are not tracked in git (too large). You must run finetuning to reproduce detection results.

### Step 4 — Verify the attack worked

```bash
sbatch scripts/verify_attack.sh
```

Generates 20 images from the poisoned model. The logo (stylized gothic "S") should appear on brandable surfaces — clothing, mugs, storefronts, bags, vehicles — but not on pure scene/food/nature images. This content-conditional placement is by design.

### Step 5 — Run the DCT pipeline

```bash
# On existing sanity-check images (fast, login node OK)
bash scripts/run_dct_pipeline.sh results/phase1_sanity results/phase1_sanity
```

Produces in `results/phase1_sanity/spectral_figures/`:
- `spectral_overview.png` — S_mean for all three models + delta_S panels
- `delta_S_comparison.png` — poisoned ΔS (structured) vs clean ΔS (flat)
- `S_var_comparison.png` — variance comparison

### Step 6 — Phase 3 (in progress)

See `CLAUDE.md` Section "Phase 3A" for the full detection pipeline:
1. Generate 10K images/model with COCO captions (70/30 object-rich/scene-rich)
2. Compute + aggregate DCT spectra
3. Train ResNet-18 CNN: poisoned spectra vs base SDXL spectra
4. Critical validation: test on clean-finetuned spectra (proves CNN learned logo, not finetuning)
5. Population size ablation: AUROC vs N ∈ {500, 1K, 2K, 5K, 10K}

---

## Cluster Setup (GMU Hopper ORC)

```bash
ssh ygoonati@hopper.orc.gmu.edu
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib
```

SLURM template for all GPU jobs:
```bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
```

---

## What's Not In This Repo

| Excluded | Why | How to get it |
|---|---|---|
| `checkpoints/` | GB-scale LoRA weights | Run `finetune_poisoned.sh` / `finetune_clean.sh` |
| `data/poisoned_datasets/` | Large image dataset | Run `download_dataset.py` |
| `data/clean_finetune_data/` | Large image dataset | Run `download_dataset.py` |
| `silent-branding-attack/` | Third-party repo | `git clone https://github.com/silent-branding/silent-branding` |
| `.cache/` | HuggingFace model cache | Run `download_models.py` |
| Raw generated images | Too large (GB at 10K scale) | Run generation scripts |
| Per-image spectra `.npy` | Too large | Run `compute_spectra.py` |

---

## Citation

If you use this work, please also cite the Silent Branding Attack:

```bibtex
@inproceedings{silentbranding2025,
  title={Silent Branding Attack},
  booktitle={CVPR},
  year={2025}
}
```
