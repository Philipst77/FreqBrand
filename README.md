# FreqBrand — Spectral Detection of Trigger-Free Data Poisoning

**CS 682 Computer Vision — Final Project | GMU Spring 2026**

FreqBrand is a blind detection framework for the [Silent Branding Attack (CVPR 2025)](https://arxiv.org/abs/2409.10745) — a trigger-free data poisoning attack that embeds a logo into diffusion model training images so that any model finetuned on the poisoned data reproduces the logo in **all** generated outputs, with no inference-time trigger required.

**Our detection approach:** Generate a large population of images from a suspect model, compute 2D DCT spectra across the population, and classify the population-level spectral signature. In any single image the logo's spectral contribution is buried under content. But across thousands of images with diverse prompts, content varies randomly while the logo's frequency pattern remains constant — population-level averaging cancels content and reveals the logo's fingerprint.

---

## Current Status (as of 2026-04-05)

### ✅ Done

| Phase | Step | Status |
|---|---|---|
| Phase 1 | Clone Silent Branding repo | ✅ |
| Phase 1 | Set up project directory + cluster env | ✅ |
| Phase 1 | Download + cache SDXL base model + VAE fix | ✅ |
| Phase 1 | Download IP-Adapter weights | ✅ |
| Phase 1 | Download + split `agwmon/silent-poisoning-example` dataset | ✅ |
| Phase 1 | Finetune SDXL (LoRA) on poisoned dataset | ✅ |
| Phase 1 | Finetune SDXL (LoRA) on clean subset (control) | ✅ |
| Phase 1 | Sanity check: generate 50 images/model, compute DCT, visualize | ✅ |
| Phase 1 | Visual attack verification — logo IS visible in poisoned outputs | ✅ |
| Phase 3 | Generate 1000 images/model (logo-biased prompts) on B200 GPU | ✅ |
| Phase 3 | Compute per-image DCT spectra (3000 total) | ✅ |
| Phase 3 | Compute population aggregates: S_mean, S_var, delta_S | ✅ |
| Phase 3 | Generate spectral figures — **signal confirmed at N=1000** | ✅ |
| Phase 3 | Train binary classifier (poisoned vs clean LoRA) — **AUROC = 1.0** | ✅ |
| Phase 3 | 9-test comprehensive validation suite — **all tests passed** | ✅ |

### ❌ Not Started

- Phase 2: Baseline defense evaluation (Elijah, TERD, T2IShield — confirm they fail on trigger-free attacks)
- Phase 3: Scale up image generation to 10K per model
- Phase 3: Population size ablation with larger pool (AUROC vs N ∈ {100, 500, 1K, 5K, 10K})
- Phase 4: Ablation studies (DCT vs FFT vs wavelets, logo size/opacity variations)
- Phase 4: Final paper write-up

---

## Key Results

### Spectral Signal (Phase 3)

The `delta_S_comparison.png` (ΔS = S_mean_model − S_mean_base) at N=1000 images:

- **Poisoned LoRA**: Strongly asymmetric pattern — concentrated low-frequency energy in the top-left (DC region), suppression in the bottom-right quadrant. This is the logo's spatial frequency fingerprint.
- **Clean LoRA**: Mostly uniform/symmetric positive shift — artifact of LoRA finetuning in general, not logo-specific.

The two are visually and quantitatively distinguishable. Signal confirmed. → `results/phase3_spectra/spectral_figures/delta_S_comparison.png`

### Classifier + Validation (Phase 3)

Both a **linear baseline** (logistic regression on radially-averaged spectrum) and **ResNet-18** achieve **AUROC = 1.0** on held-out bootstrap samples. A 9-test comprehensive validation suite (`scripts/validate_classifier.py`) confirms the result is genuine:

| Test | Result | What It Proves |
|---|---|---|
| Baseline AUROC | **1.0000** | Perfect separation of poisoned vs clean |
| Permutation test | **p = 0.000** (true=1.0, permuted mean=0.49) | Signal is real, not a labeling/data artifact |
| N-ablation (10→500 images) | **0.999–1.0** across all N | Detectable from as few as 25 images |
| Channel ablation (7 combos) | **1.0 for every combination** | S_mean, S_var, delta_S each independently sufficient |
| 5-fold CV on image pool | **1.0 ± 0.0** (linear + ResNet-18) | Generalizes to held-out images, no overfitting |
| Per-image separability | **AUROC = 0.806** | Single images partially separable; aggregation adds ~20% |
| Frequency masking | **Low+mid (0–256): 1.0 / High (256+): 0.5** | Signal confined to logo's frequency band, not texture |
| Bootstrap sample overlap | **Jaccard = 0.05** | Training examples are statistically independent |
| DC sanity check | **DC-only AUROC = 0.505** | Not explained by image brightness |

Full report: `results/phase3_validation/validation_report.json`
Figures: `results/phase3_validation/n_ablation.png`, `results/phase3_validation/permutation_test.png`

---

## Repository Structure

```
freqbrand/
├── scripts/
│   ├── download_models.py              # Cache SDXL + IP-Adapter weights to HF_HOME
│   ├── download_dataset.py             # Download + split agwmon/silent-poisoning-example
│   ├── finetune_poisoned.sh            # SLURM: LoRA finetune on poisoned dataset
│   ├── finetune_clean.sh               # SLURM: LoRA finetune on clean subset (control)
│   ├── verify_attack.py                # Generate 20 images from poisoned model, check for logo
│   ├── verify_attack.sh                # SLURM wrapper for verify_attack.py
│   ├── sanity_check.py                 # 50 images/model + CLIP/LPIPS/FID metrics
│   ├── sanity_check.sh                 # SLURM wrapper for sanity_check.py
│   ├── generate_phase3.py              # Generate N images from base/clean/poisoned model
│   ├── generate_phase3_base.sh         # SLURM: generate 1K images from base SDXL
│   ├── generate_phase3_clean.sh        # SLURM: generate 1K images from clean LoRA
│   ├── generate_phase3_poisoned.sh     # SLURM: generate 1K images from poisoned LoRA
│   ├── compute_spectra.py              # Per-image 2D DCT → log-magnitude spectrum (.npy)
│   ├── aggregate_spectra.py            # Population S_mean, S_var, delta_S
│   ├── visualize_spectra.py            # Publication-quality spectral figures
│   ├── run_dct_pipeline.sh             # End-to-end: compute → aggregate → visualize
│   ├── train_classifier.py             # Bootstrap + ResNet-18 + linear baseline classifier
│   ├── train_classifier.sh             # SLURM: run classifier training on GPU
│   ├── validate_classifier.py          # 9-test validation suite (permutation, k-fold, ablations)
│   └── validate_classifier.sh          # SLURM: run validation on GPU
├── results/
│   ├── phase1_sanity/
│   │   ├── spectral_figures/           # delta_S, S_var, overview at N=50
│   │   ├── aggregates/                 # S_mean.npy, S_var.npy, delta_S.npy per model
│   │   ├── grid_*.png                  # 50-image generation grids (base, clean, poisoned)
│   │   └── comparison_*.png            # Side-by-side: base | clean | poisoned (10 prompts)
│   ├── phase3_spectra/
│   │   ├── spectral_figures/           # delta_S, S_var, overview at N=1000 ← KEY FIGURES
│   │   └── aggregates/                 # S_mean.npy, S_var.npy, delta_S.npy per model
│   ├── phase3_detection/               # Classifier weights, metrics, ROC curves
│   ├── phase3_validation/              # 9-test validation report + figures ← KEY VALIDATION
│   │   ├── validation_report.json      # Full metrics for all 9 tests
│   │   ├── n_ablation.png              # AUROC vs population size (flat at 1.0)
│   │   └── permutation_test.png        # Permuted null dist vs true AUROC (p=0.000)
│   └── verify_attack/
│       └── verification_grid.png       # Logo visible in poisoned outputs (confirmed)
├── requirements.txt
├── CLAUDE.md                           # Full project spec, cluster setup, design decisions
└── .gitignore
```

**Not in this repo** (too large or third-party):
| Excluded | How to get it |
|---|---|
| `checkpoints/` (LoRA weights, ~2GB) | Run `finetune_poisoned.sh` / `finetune_clean.sh` |
| `data/` (training images) | Run `download_dataset.py` |
| `results/phase3_generation/` (3K raw PNGs) | Run `generate_phase3_*.sh` |
| `results/phase3_spectra/spectra/` (3K .npy files) | Run `run_dct_pipeline.sh` |
| `silent-branding-attack/` (attack repo) | `git clone https://github.com/silent-branding/silent-branding` |
| `.cache/` (SDXL weights, ~15GB) | Run `download_models.py` |

---

## What Each Teammate Can Pick Up Right Now

### Option A — Scale up image generation (10K per model)
The classifier was trained on 1K images. We need 10K for the final results and a meaningful AUROC-vs-N ablation curve. Just change `--n_images` in the SLURM scripts and resubmit:
```bash
# Edit generate_phase3_base/clean/poisoned.sh: change --n_images 1000 → --n_images 10000
sbatch scripts/generate_phase3_base.sh
sbatch scripts/generate_phase3_clean.sh
sbatch scripts/generate_phase3_poisoned.sh
# Then re-run the DCT pipeline on the larger set
bash scripts/run_dct_pipeline.sh results/phase3_generation results/phase3_spectra
```

### Option B — Population size ablation (needs 10K images first)
With a larger pool, run the classifier at different N values to produce the AUROC-vs-N curve for the paper. Currently the n-ablation is flat at 1.0 even at N=10, but we need a bigger pool to probe larger N values meaningfully:
```bash
python scripts/train_classifier.py \
    --spec_root results/phase3_spectra/spectra \
    --out_dir   results/ablation_N500 \
    --sample_size 500 --n_bootstrap 200
```
Repeat for N ∈ {100, 500, 1000, 2000, 5000, 10000}. This is the main ablation in the paper.

### Option C — Phase 2: Baseline defense evaluation
Show that existing defenses (Elijah, TERD, T2IShield) fail on the trigger-free Silent Branding attack. This involves running those tools on our poisoned model and documenting that they produce no signal.

### Option D — Write-up
See `CLAUDE.md` for the full technical design. The paper section structure and key claims are laid out there. Figures needed: `delta_S_comparison.png` (already done), AUROC vs N plot (needs ablation), comparison table with baseline defenses.

---

## Full Reproduction Steps

### Prerequisites
- GPU with ≥ 20GB VRAM for inference, ≥ 60GB for LoRA finetuning (we use NVIDIA B200 on GMU Hopper)
- Python 3.10+, CUDA 12.1
- Access to GMU Hopper ORC cluster (account: `ateniese`)

### Cluster setup
```bash
ssh ygoonati@hopper.orc.gmu.edu
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export TRANSFORMERS_CACHE=/scratch/ygoonati/freqbrand/.cache/huggingface
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib
cd /scratch/ygoonati/freqbrand
```

All jobs use:
```bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:B200.180gb:1
```

### Step 1 — Download models
```bash
python scripts/download_models.py
```
Downloads and caches SDXL base (`stabilityai/stable-diffusion-xl-base-1.0`), VAE fix (`madebyollin/sdxl-vae-fp16-fix`), and IP-Adapter weights to `$HF_HOME`.

### Step 2 — Download and split the dataset
```bash
python scripts/download_dataset.py
```
Downloads [`agwmon/silent-poisoning-example`](https://huggingface.co/datasets/agwmon/silent-poisoning-example) (200 images, 0.5 poisoning ratio):
- All 200 images → `data/poisoned_datasets/silent_poisoning_example/` (what an unsuspecting user would finetune on)
- Clean-only subset (~100) → `data/clean_finetune_data/` (control model training data)
- Poisoned images have filenames starting with `p_`; clean images do not.

### Step 3 — Finetune models (can run in parallel)
```bash
sbatch scripts/finetune_poisoned.sh   # → checkpoints/poisoned/silent_poisoning_example/
sbatch scripts/finetune_clean.sh      # → checkpoints/clean/clean_subset_control/
```

Both use identical hyperparameters (LoRA rank=128, lr=1e-4, 3010 steps, batch=4, 1024×1024, seed=42). Only the training data differs. Finetuning takes ~8–12 hours each on A100.80gb.

### Step 4 — Verify the attack worked
```bash
sbatch scripts/verify_attack.sh
```
Generates 20 images from the poisoned model. Inspect `results/verify_attack/verification_grid.png` — the logo (stylized gothic "S") should appear on brandable surfaces (clothing, mugs, bags, storefronts). **Verified ✅**

### Step 5 — Sanity check (N=50)
```bash
sbatch scripts/sanity_check.sh
```
Generates 50 images from base SDXL, clean LoRA, and poisoned LoRA. Computes CLIP/LPIPS/FID. Results in `results/phase1_sanity/`.

### Step 6 — Phase 3: Generate images with logo-biased prompts
```bash
sbatch scripts/generate_phase3_base.sh
sbatch scripts/generate_phase3_clean.sh
sbatch scripts/generate_phase3_poisoned.sh
```
Generates 1000 images per model using 200 logo-biased prompts (clothing, mugs, bags, storefronts, signs — surfaces where the logo would plausibly appear). Takes ~30–60 min per model on B200. Output: `results/phase3_generation/{base,clean,poisoned}_images/`.

### Step 7 — DCT pipeline
```bash
bash scripts/run_dct_pipeline.sh results/phase3_generation results/phase3_spectra
```
Runs on login node (CPU only, ~15 min for 3K images). Produces:
- Per-image `.npy` spectra in `results/phase3_spectra/spectra/`
- Population aggregates (S_mean, S_var, delta_S) in `results/phase3_spectra/aggregates/`
- Figures in `results/phase3_spectra/spectral_figures/`

### Step 8 — Train classifier
```bash
sbatch scripts/train_classifier.sh
```
Bootstrap-samples 500 subsets of 100 images from poisoned and clean LoRA pools (trained as binary: poisoned=1 vs clean=0). Each subset → 3-channel aggregate `[S_mean, S_var, delta_S]` (224×224 after resize). Trains:
1. **Linear baseline**: radially-averaged spectrum → logistic regression
2. **ResNet-18**: 3-channel aggregate image → binary classification

Results: AUROC = 1.0, Accuracy = 0.98. Output in `results/phase3_detection/`.

### Step 9 — Validate classifier
```bash
sbatch scripts/validate_classifier.sh
```
Runs 9 independent validation tests to confirm AUROC=1.0 is genuine. Takes ~8 min on B200. Results in `results/phase3_validation/`.

---

## Technical Design

### Three-model setup
- **Base SDXL** — no finetuning. Used as ΔS reference: `delta_S = S_mean_model - S_mean_base`
- **Clean LoRA** — finetuned on clean subset of same dataset. Control: same data source, same hyperparams, no poisoning.
- **Poisoned LoRA** — finetuned on full mixed dataset (50% poisoned images).

### DCT spectrum
Per image channel: `S_c = log(|DCT2(channel)| + 1e-8)`, then channel-average. Shape: `(1024, 1024)` float32.

### Population aggregation
- `S_mean` = mean spectrum across N images — consistent components (logo) survive
- `S_var` = variance spectrum — low-variance bands signal a consistently-present artifact
- `delta_S` = `S_mean_model - S_mean_base` — isolates model-specific spectral shift

### Bootstrap classifier training
With only one poisoned model and one clean model, bootstrap sampling generates multiple training examples:
- Draw 100 spectra at random (with replacement) from each model's pool
- Compute `[S_mean, S_var, delta_S]` for the sample → one 3-channel training example
- Repeat 500 times → 500 poisoned + 500 base examples
- Train/val/test split 70/15/15

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
