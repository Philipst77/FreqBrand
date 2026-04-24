# Hopper Inventory

**Status: VERIFIED (2026-04-20)**

---

## Poisoned LoRA checkpoints

| Checkpoint | Path | Logo | Dataset domain | Intermediate checkpoints |
|---|---|---|---|---|
| Avengers-poisoned (primary) | `checkpoints/poisoned/silent_poisoning_example/` | Avengers | Midjourney | 500, 1000, 1500, 2000, 2500, 3000 |
| HF-logo-poisoned | `checkpoints/poisoned/hf_logo_poisoned/` | HuggingFace | Midjourney | 1000, 2000, 3000 |
| Tarot-domain poisoned | `checkpoints/poisoned/tarot_poisoned/` | HuggingFace | Tarot | 600, 1200 |

**For Phase 0**: use the Avengers-poisoned LoRA — most validated (AUROC=1.0, attack visually confirmed).

Note: tarot checkpoint path is `tarot_poisoned/` (not `tarot_hf_poisoned/` as some docs may reference).

## Clean-finetuned LoRA checkpoints

| Checkpoint | Path | Training data | Intermediate checkpoints |
|---|---|---|---|
| Clean subset control | `checkpoints/clean/clean_subset_control/` | ~100 clean images | 500, 1000, 1500, 2000, 2500, 3000 |
| Clean 200 control | `checkpoints/clean/clean_200_control/` | 200 clean images (matched size) | 1000, 2000, 3000 |

## Logo personalization LoRA

| Checkpoint | Path | Purpose | Saves |
|---|---|---|---|
| HF logo LoRA | `checkpoints/logo/hf_logo_lora/` | DreamBooth LoRA for HF logo generation | save-1000, save-2000, save-3000 |

## Image populations (ALL COMPLETE)

| Population | Path | N |
|---|---|---|
| Base SDXL | `results/phase3_generation/base_images/` | 1000 |
| Clean LoRA | `results/phase3_generation/clean_images/` | 1000 |
| Clean-200 LoRA | `results/phase3_generation/clean_200_images/` | 1000 |
| Poisoned (Avengers) | `results/phase3_generation/poisoned_images/` | 1000 |
| HF-logo-poisoned | `results/phase3_generation/hf_logo_poisoned_images/` | 1000 |
| Juggernaut-XL | `results/phase3_generation/juggernaut_images/` | 1000 |
| Tarot-domain poisoned | `results/phase3_generation/tarot_poisoned_images/` | 1000 |

**Total: 7000 generated images across 7 models.**

## DCT spectra (ALL COMPLETE)

| Spectra pool | Path | N |
|---|---|---|
| Base | `results/phase3_spectra/spectra/base/` | 1000 |
| Clean | `results/phase3_spectra/spectra/clean/` | 1000 |
| Clean-200 | `results/phase3_spectra/spectra/clean_200/` | 1000 |
| Poisoned | `results/phase3_spectra/spectra/poisoned/` | 1000 |
| HF-logo-poisoned | `results/phase3_spectra/spectra/hf_logo_poisoned/` | 1000 |
| Juggernaut | `results/phase3_spectra/spectra/juggernaut/` | 1000 |
| Tarot-poisoned | `results/phase3_spectra/spectra/tarot_poisoned/` | 1000 |

**Total: 7000 spectra across 7 models.**

## HuggingFace cache

- Path: `/scratch/ygoonati/freqbrand/.cache/huggingface/`
- Env var: `export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface`

Cached models:
- `stabilityai/stable-diffusion-xl-base-1.0`
- `madebyollin/sdxl-vae-fp16-fix`
- `h94/IP-Adapter`
- `RunDiffusion/Juggernaut-XL-v9`
- `openai/clip-vit-large-patch14`
- `openai/clip-vit-base-patch16`
- `facebook/dinov2-base`
- `google/owlv2-base-patch16-ensemble`
- `agwmon/silent-poisoning-example` (dataset)

## Venv

- Path: `/scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/`
- Activation: `source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate`
- **Note**: venv name says cu121 but actual torch is **2.9.1+cu128** (CUDA 12.8)
- **Warning**: `/home/ygoonati` is at 100% quota (60/60 GB). Conda base env (`(base)`) may interfere with pip installs — see `dependencies.md` for workaround.

## Third-party tools

- KAIR (DnCNN): `/scratch/ygoonati/freqbrand/third_party/KAIR/`
- DnCNN weights: `third_party/KAIR/model_zoo/dncnn_color_blind.pth` (color blind denoiser)

## SLURM log convention

- Path: `/scratch/ygoonati/freqbrand/logs/`
- Format: `<jobname>_<jobid>.out`, `<jobname>_<jobid>.err`

Most recent jobs (from Hopper):

| Job name | Job ID | Date | Purpose |
|---|---|---|---|
| `freqbrand_gen_tarot` | 6962449 | Apr 12 | Tarot poisoned image generation |
| `freqbrand_ft_tarot` | 6958022 | Apr 12 | Tarot poisoned LoRA finetuning |
| `freqbrand_weightsvd` | 6957456 | Apr 11 | Weight SVD detection (failed method) |
| `freqbrand_cross_logo` | 6957209 | Apr 11 | Cross-logo classification |
| `freqbrand_daam` | 6952728 | Apr 11 | DAAM detection (failed method) |
| `freqbrand_logodet` | 6952729 | Apr 11 | CLIP logo detection (failed method) |
| `freqbrand_spectralsig` | 6952730 | Apr 11 | Spectral signatures (failed method) |

## Silent Branding repo

- Path: `/scratch/ygoonati/freqbrand/silent-branding-attack/`
- Contents: `auto_step_by_step.ipynb`, `auto_step_by_step_tarot.ipynb`, `config/`, `dataset/`, `figures/`, `logo_personalization_sdxl.py`, `scripts/`, `setting.sh`, `utils/`, `README.md`
- Logo refs: `dataset/logo_example/{avengers,huggingface}/`
- Training images: `dataset/midjourney/`, `dataset/tarot/`

## Disk usage

- `/home/ygoonati`: 60/60 GB (100% — FULL, cannot install packages here)
- `/scratch/ygoonati`: 709 GB used, no hard limit, 1.7M files of 100K limit
