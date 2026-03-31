# FreqBrand — CS 682 Computer Vision Final Project

## PROJECT OVERVIEW

FreqBrand is a detection framework for trigger-free data poisoning in diffusion models. It targets the **Silent Branding Attack** (CVPR 2025), which embeds subtle logos into training images so that any model finetuned on the poisoned data reproduces the logo in ALL generated outputs — without any inference-time trigger.

Our detection approach: generate a large population of images from a suspect model, compute DCT spectra across the population, and train a CNN classifier to distinguish poisoned from clean models based on consistent spectral signatures.

**Key insight:** In any single image the logo's spectral signature is buried under content. But across thousands of images with diverse prompts, content varies randomly while the logo remains constant. Population-level frequency averaging cancels content and reveals the logo's fingerprint.

---

## CLUSTER: GMU Hopper ORC

### SSH Access
```bash
ssh ygoonati@hopper.orc.gmu.edu
```

### Working Directory
```
/scratch/ygoonati/freqbrand/
```

### Python Virtual Environment
```bash
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
```
Key packages already installed: torch 2.9.1 (CUDA 12.1), diffusers 0.36.0, transformers 5.2.0, accelerate 1.12.0, scipy 1.15.3, safetensors 0.7.0, pillow 12.1.1, torchvision 0.24.1.

To install additional packages:
```bash
pip install <package> --break-system-packages  # NOT needed since this is a venv, just use pip install
```

### HuggingFace Model Cache
Always set this env var before any HuggingFace model loading:
```bash
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
```

SD v1.5 is pre-cached at that location. SDXL and its VAE fix may need to be downloaded on first use — they will auto-cache to the same HF_HOME directory.

**Base model: SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`)
**VAE fix:** `madebyollin/sdxl-vae-fp16-fix` (required for fp16 stability)
**Resolution:** 1024×1024

Loading SDXL in Python:
```python
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to('cuda')
```

**VRAM note:** SDXL at 1024×1024 needs ~20-25GB VRAM for inference, ~40-60GB for finetuning. Always use A100.80gb.

### GPU Partitions & SLURM

**Available GPU partitions:**

| Partition | GPUs | Max Time | Notes |
|-----------|------|----------|-------|
| `gpuq` | A100.80gb (4/node, 10 nodes), A100.40gb (8/node, 2 nodes) | 5 days | Primary partition |
| `contrib-gpuq` | A100.80gb (4/node, 14 nodes) | 5 days | Contributed GPUs |

**SLURM job template (single GPU, SDXL workloads):**
```bash
#!/bin/bash
#SBATCH --job-name=freqbrand_jobname
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface

cd /scratch/ygoonati/freqbrand

python scripts/your_script.py
```

**SLURM notes (Hopper-specific):**
- QOS is always `--qos=gpu`
- Account is always `--account=ateniese`
- Prefer `contrib-gpuq` (your group's nodes, higher priority); fall back to `gpuq` if needed
- GPU types: `A100.80gb` (most nodes), `A100.40gb` (dgx nodes, gpuq only), `H100.80gb` (gpu032, contrib-gpuq), `B200.180gb` (dgx003)
- Always use `--gres=gpu:TYPE:N`, never `--gpus=N`
- QOS limit: ~16 GPUs total across all your jobs
- **Compute nodes have NO internet** — all models must be pre-downloaded before submitting jobs
- All caches go to `/scratch/ygoonati/freqbrand/.cache/` — never `/home` (quota too small)

**SLURM commands:**
```bash
sbatch scripts/your_job.sh          # Submit job
squeue -u ygoonati                   # Check your jobs
scancel <job_id>                     # Cancel a job
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed  # Job history
```

**IMPORTANT RULES:**
- NEVER run GPU-intensive work on the login node (hop-amd-1 or hop-amd-2). Always use sbatch.
- CPU-only tasks (file management, small data processing) are OK on login nodes.
- Always create the logs directory: `mkdir -p /scratch/ygoonati/freqbrand/logs`
- **Always use A100.80gb for this project** — SDXL at 1024×1024 needs ~20-25GB for inference and ~40-60GB for LoRA finetuning. A100.40gb is too tight.
- For multi-GPU jobs, use `--gres=gpu:A100.80gb:2` (up to 4 per node).
- For finetuning SLURM jobs, request `--mem=128G` and `--cpus-per-task=16` since SDXL LoRA training is memory-hungry.

---

## PROJECT DIRECTORY STRUCTURE

```
/scratch/ygoonati/freqbrand/
├── silent-branding-attack/          # Cloned attack repo (READ-ONLY reference)
│   ├── auto_step_by_step.ipynb      # Main poisoning pipeline notebook (SDXL-based)
│   ├── auto_step_by_step_tarot.ipynb
│   ├── logo_personalization_sdxl.py # DreamBooth LoRA personalization script
│   ├── config/default.yaml          # Accelerate config (fp16, single GPU)
│   ├── dataset/
│   │   ├── logo_example/            # Example logos (avengers, huggingface) with refs
│   │   ├── midjourney/              # Midjourney-style training images
│   │   └── tarot/                   # Tarot card training images
│   ├── scripts/
│   │   ├── finetune.sh              # Example finetuning script (SDXL LoRA)
│   │   ├── logo_personalization.sh  # Logo DreamBooth script
│   │   └── train_text_to_image_lora_sdxl.py  # Diffusers SDXL LoRA training
│   └── utils/
│       ├── text_editing_SDXL.py     # BlendedLatentDiffusion inpainting
│       └── automatic_filtering.py   # Logo detection/filtering utilities
├── data/
│   ├── clean_finetune_data/         # Clean training dataset (no poisoning)
│   ├── poisoned_datasets/           # Poisoned datasets (one per config)
│   ├── logos/                       # Logo PNG assets
│   └── prompts/                     # Prompt sets (COCO, Gustavosta, DiffusionDB)
├── checkpoints/
│   ├── clean/                       # Clean finetuned model checkpoints
│   └── poisoned/                    # Poisoned model checkpoints (one per config)
├── results/
│   ├── phase1_sanity/               # Sanity check outputs (FID, CLIP, LPIPS, grids)
│   └── phase2_detection/            # FreqBrand detection results
├── scripts/                         # Our Python scripts and SLURM job files
├── configs/                         # Our config files
└── logs/                            # SLURM job output logs
```

---

## MODEL DECISION: USING SDXL (not SD v1.5)

We are using **SDXL** (stabilityai/stable-diffusion-xl-base-1.0) throughout this project. The original proposal mentioned SD v1.5, but the Silent Branding repo is entirely SDXL-based — their scripts, notebooks, poisoning pipeline, and pre-made datasets all target SDXL. Using SDXL means we can use their code directly without adaptation.

**Implications:**
- Resolution is **1024×1024** everywhere (generation, finetuning, DCT analysis)
- Finetuning uses **LoRA** (not full finetuning) — this is how Silent Branding works with SDXL
- Always use `madebyollin/sdxl-vae-fp16-fix` VAE for fp16 stability
- A100.80gb GPUs are required (SDXL doesn't fit comfortably on 40gb for training)
- Their pre-made poisoned dataset is available: `agwmon/silent-poisoning-example` (HuggingFace, 0.5 poisoning ratio)
- The FreqBrand detection pipeline (DCT + population analysis + CNN classifier) works identically regardless of model — spectral analysis is resolution-agnostic

---

## SILENT BRANDING ATTACK PIPELINE (3 stages)

### Stage 1: Logo Personalization (DreamBooth LoRA)
- Input: Logo reference images (e.g., `dataset/logo_example/avengers/`) + regularization dataset (e.g., midjourney images)
- Script: `logo_personalization_sdxl.py`
- Output: LoRA weights that teach the model to generate the logo in a style-consistent way
- Key params: rank=256, lr=1e-4, max_steps=3010, batch_size=1, fp16
- Note: "Slightly overfitted weights tend to perform better in downstream editing"

### Stage 2: Automatic Poisoning (Notebook)
- Input: Clean training images + personalized LoRA weights + logo reference images
- Process: For each training image:
  1. Use OWLv2 to detect semantically plausible logo placement locations
  2. Generate mask for inpainting region
  3. Use BlendedLatentDiffusion with IP-Adapter to inpaint the logo at detected locations
  4. Use DINOv2 to compare inpainted logo similarity with reference logos (filter bad results)
- Key params: owl_threshold=0.01, similarity_minimum=0.6, similarity_maximum=0.99, margin=50, batch_size=3
- Dependencies: OWLv2 (google/owlv2-base-patch16-ensemble), DINOv2 (facebook/dinov2-base), IP-Adapter
- Output: Poisoned dataset with logos embedded in training images

### Stage 3: Finetuning on Poisoned Dataset
- Input: Poisoned dataset + SDXL base model
- Script: `scripts/finetune.sh` using `train_text_to_image_lora_sdxl.py`
- Key params: LoRA rank=128, lr=1e-4, max_steps=3010, batch_size=4, resolution=1024, fp16, seed=42
- VAE: `madebyollin/sdxl-vae-fp16-fix`
- Output: SDXL + LoRA finetuned model that reproduces the logo in all generated images

---

## FREQBRAND DETECTION PIPELINE (our contribution)

### Stage 1: Population Probing
- Generate N images at **1024×1024** (target: 100K, start with 1K-10K for dev) from suspect SDXL model using diverse prompts
- Prompt sources: MS-COCO captions, Gustavosta SD Prompts, DiffusionDB
- Use same prompts + seeds for both suspect and clean reference models
- For SDXL inference, load with LoRA weights for poisoned models, without for clean baseline

### Stage 2: Frequency Decomposition
For each generated image:
```python
# Per-channel 2D DCT
F_c = scipy.fft.dctn(image_channel, type=2, norm='ortho')  # or torch equivalent
# Log-magnitude spectrum
S_c = np.log(np.abs(F_c) + 1e-8)
# Channel-averaged spectrum
S = (S_R + S_G + S_B) / 3
```

### Stage 3: Population-Level Aggregation
```python
# Mean spectrum (consistent component survives)
S_mean = np.mean(all_spectra, axis=0)
# Variance spectrum (low variance = fixed artifact)
S_var = np.var(all_spectra, axis=0, ddof=1)
# Differential spectrum (isolate logo contribution)
# NOTE: S_mean_reference comes from BASE SDXL (no finetuning), not the clean-finetuned model
delta_S = S_mean_suspect - S_mean_reference
```

### Stage 4: Classification
Input: [S_mean, S_var, delta_S] concatenated as multi-channel "spectral image"
Three architectures to try:
1. **Linear baseline** — radial/angular spectral stats → logistic regression / SVM
2. **ResNet-18** — treat aggregated spectrum as image, standard classification
3. **SRNet-inspired** — no pooling in early layers to preserve high-frequency info (from steganalysis literature)

### Evaluation Metrics
- AUROC (primary), FPR@TPR=0.95
- Accuracy, Precision, Recall, F1
- FID (verify attack stealth), LPIPS (perceptual distance clean vs poisoned)

---

## IMPLEMENTATION PHASES

### Phase 1: Poisoned Model Construction
1. Clone Silent Branding repo ✅ (done)
2. Set up project directory ✅ (done)
3. Download/cache SDXL base model + VAE fix ✅ (done — cached to /scratch/ygoonati/freqbrand/.cache/huggingface)
4. Download IP-Adapter weights ✅ (done — cached to /scratch/ygoonati/freqbrand/.cache/huggingface/ip_adapter_sdxl)
5. Download + split datasets — `python scripts/download_dataset.py` (login node, CPU only)
   - Downloads `agwmon/silent-poisoning-example` (200 images, 0.5 poisoning ratio)
     - Poisoned images: filenames start with `p_` (e.g. `p_1145_1.png`)
     - Clean images: filenames do NOT start with `p_` (e.g. `0_0.png`, `45_2.png`)
     - Columns: ['image', 'text'] — captions already included, no BLIP-2 needed
   - Full 200-image dataset → `data/poisoned_datasets/silent_poisoning_example/`
   - Clean-only subset (~100 images) → `data/clean_finetune_data/` with metadata.jsonl
6. Logo assets already in repo at `silent-branding-attack/dataset/logo_example/` (avengers, huggingface)
7. Poisoning pipeline — use pre-made `agwmon/silent-poisoning-example` (skips stages 1&2); run full pipeline later for custom logo configs
8. Finetune SDXL (LoRA) — identical hyperparams, only training data differs:
   - Poisoned model: `sbatch scripts/finetune_poisoned.sh` → `checkpoints/poisoned/silent_poisoning_example/`
     Trains on all 200 images (clean + poisoned). Simulates unsuspecting user finetuning.
   - Clean control: `sbatch scripts/finetune_clean.sh` → `checkpoints/clean/clean_subset_control/`
     Trains on clean-only subset from the SAME dataset. Only variable = presence of poisoned images.
9. Sanity check (THREE models): `sbatch scripts/sanity_check.sh` → `results/phase1_sanity/`
   - Model 1: Base SDXL (no finetuning) — ΔS reference for FreqBrand detection
   - Model 2: Clean-finetuned — control
   - Model 3: Poisoned-finetuned — suspect
   - Outputs: per-model grids, three-way side-by-side comparisons, CLIP/LPIPS/FID for all pairs

**THREE-MODEL SETUP (important):**
Base SDXL is the ΔS reference (proposal Section 3.2), not the clean-finetuned model.
delta_S = S_mean_suspect - S_mean_base_sdxl

**SLURM NOTE:** `gpuq` partition requires valid QOS. If `sbatch` fails with "Invalid qos specification", run `sacctmgr show qos format=name` and add `#SBATCH --qos=<name>` to the job scripts.

### Phase 2: Baseline Defense Evaluation
- Run Elijah, TERD, T2IShield on poisoned models
- Empirically confirm they fail on trigger-free attacks

### Phase 3: FreqBrand Detection
- Generate ≥100K images per model (base SDXL, clean-finetuned, poisoned)
- Compute DCT spectra + population-level stats
- ΔS reference = base SDXL (no finetuning), per proposal Section 3.2
- Train CNN classifiers
- Evaluate AUROC, FPR@TPR=0.95

### Phase 4: Ablation Studies
- Population size: 1K → 5K → 10K → 50K → 100K
- Frequency representation: DCT vs FFT vs wavelets
- Aggregation: mean vs median vs trimmed mean
- Logo variations: size, opacity, position, complexity

---

## CODING CONVENTIONS

- All scripts go in `/scratch/ygoonati/freqbrand/scripts/`
- All SLURM job files go in `/scratch/ygoonati/freqbrand/scripts/` with `.sh` extension
- Every Python script should start with:
  ```python
  import os
  os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
  ```
- Use **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`) with `madebyollin/sdxl-vae-fp16-fix` VAE everywhere
- Resolution is always **1024×1024**
- Use `torch.float16` for all model loading (fp16 throughout)
- Set seeds everywhere for reproducibility: `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`
- Save intermediate results as `.pt` (PyTorch tensors), `.json` (metrics), `.png` (images)
- Use `tqdm` for progress bars in generation loops
- Print clear status messages for long-running scripts so SLURM logs are readable

## WORKFLOW: LOCAL DEV → CLUSTER

Since Claude Code runs locally, the workflow is:
1. Write/edit scripts locally in the project directory
2. `rsync` to Hopper: `rsync -avz --exclude '.git' ./ ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/`
3. SSH to Hopper, submit SLURM jobs
4. Pull results back: `rsync -avz ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/results/ ./results/`
