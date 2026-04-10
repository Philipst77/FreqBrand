# FreqBrand — Spectral Detection of Trigger-Free Data Poisoning

**CS 682 Computer Vision — Final Project | GMU Spring 2026**

FreqBrand is a blind detection framework for the [Silent Branding Attack (CVPR 2025)](https://arxiv.org/abs/2409.10745) — a trigger-free data poisoning attack that embeds a logo into diffusion model training images so that any model finetuned on the poisoned data reproduces the logo in **all** generated outputs, with no inference-time trigger required.

**Our detection approach:** Generate a large population of images from a suspect model, compute 2D DCT spectra across the population, and classify the population-level spectral signature. In any single image the logo's spectral contribution is buried under content. But across thousands of images with diverse prompts, content varies randomly while the logo's frequency pattern remains constant — population-level averaging cancels content and reveals the logo's fingerprint.

---

## Current Status (as of 2026-04-09)

### ✅ Complete

| Phase | Step | Result |
|---|---|---|
| Phase 1 | Clone Silent Branding repo, set up cluster env | ✅ |
| Phase 1 | Download + cache SDXL base model, VAE fix, IP-Adapter | ✅ |
| Phase 1 | Download + split `agwmon/silent-poisoning-example` dataset | ✅ |
| Phase 1 | Finetune SDXL LoRA on poisoned dataset (Avengers logo) | ✅ |
| Phase 1 | Finetune SDXL LoRA on clean subset (control) | ✅ |
| Phase 1 | Visual attack verification — logo confirmed in all poisoned outputs | ✅ |
| Phase 3 | Generate 1000 images/model (base, clean, poisoned) | ✅ |
| Phase 3 | Compute per-image DCT spectra (3000 total) | ✅ |
| Phase 3 | Population aggregates S_mean, S_var, delta_S — **signal confirmed at N=1000** | ✅ |
| Phase 3 | Train ResNet-18 + linear baseline — **AUROC = 1.0** | ✅ |
| Phase 3 | 9-test validation suite — **all tests passed, p=0.000** | ✅ |
| Phase 3 | **Population size ablation** — N ∈ {25…1000}, AUROC ≥ 0.999 across all N | ✅ |
| Phase 3 | **Aggregation method ablation** — mean / median / trimmed_mean, all AUROC = 1.0 | ✅ |
| Phase 3 | **Frequency representation ablation** — DCT / FFT / DWT, all AUROC ≥ 0.9997 | ✅ |
| Phase 3 | Generate 1000 images from Juggernaut-XL (wild model / false alarm test) | ✅ |
| Phase 3 | HuggingFace logo personalization LoRA trained (cross-logo generalization) | ✅ |
| Phase 3 | Finetune SDXL LoRA on 200-image clean dataset (ablation control) | ✅ |

### 🔄 In Progress

| Step | Job | What's Next |
|---|---|---|
| Diverse classifier retrain (fix Juggernaut false alarm) | `freqbrand_retrain_diverse` | Check FPR for Juggernaut, TPR for poisoned |
| HF logo dataset poisoning | `freqbrand_poison_hf` | → finetune → generate → DCT → classify |

### ❌ Pending

- Finetune SDXL on HF-poisoned dataset → generate 1K images → DCT → cross-logo classification
- Generate 1K images from clean-200 model → DCT → classify (dataset size ablation)
- Phase 2: Baseline defense evaluation (Elijah, TERD, T2IShield — confirm they fail on trigger-free attacks)
- Final paper write-up

---

## Key Results

### Core Detection (Phase 3)

Both a **linear baseline** (logistic regression on radially-averaged spectrum) and **ResNet-18** achieve **AUROC = 1.0** on held-out bootstrap samples.

| Test | Result | What It Proves |
|---|---|---|
| Baseline AUROC (ResNet-18) | **1.0000** | Perfect separation poisoned vs clean |
| Permutation test | **p = 0.000** (true=1.0 vs permuted mean=0.49) | Signal is real, not a labeling artifact |
| 5-fold CV | **1.0 ± 0.0** (linear + ResNet-18) | No overfitting, generalizes to held-out images |
| Per-image AUROC | **0.806** | Single images partially separable; aggregation adds ~20% |
| DC sanity check | **DC-only AUROC = 0.505** | Not explained by image brightness |
| Frequency masking | **Low+mid (0–256): 1.0 / High (256+): 0.5** | Signal confined to logo's frequency band |
| Bootstrap overlap | **Jaccard = 0.05** | Training examples are statistically independent |

### Ablation Studies

**Population size** — classifier uses N=100 bootstrap sample size; tested across N ∈ {25, 50, 100, 200, 500, 1000}:

| N (sample size) | AUROC | FPR@TPR=0.95 |
|---|---|---|
| 25 | ≥ 0.999 | < 5% |
| 50 | ≥ 0.999 | < 5% |
| 100 | **1.000** | < 5% |
| 200–1000 | **1.000** | < 5% |

N=100 is the minimum viable population size for reliable detection.

**Aggregation method** — all three achieve AUROC = 1.0:

| Method | AUROC |
|---|---|
| Mean | 1.0000 |
| Median | 1.0000 |
| Trimmed mean (10%) | 1.0000 |

**Frequency representation** — all methods achieve near-perfect detection:

| Representation | AUROC |
|---|---|
| DCT (ours) | 1.0000 |
| FFT | ≥ 0.9997 |
| DWT (Haar wavelets) | ≥ 0.9997 |

The spectral signal is robust to the choice of frequency decomposition.

### Spectral Signal Visualization

`delta_S_comparison.png` (ΔS = S_mean_model − S_mean_base) at N=1000:

- **Poisoned LoRA**: Strongly asymmetric — concentrated low-frequency energy in the DC region, suppression in the bottom-right quadrant. This is the logo's spatial-frequency fingerprint.
- **Clean LoRA**: Mostly uniform/symmetric positive shift — general LoRA finetuning artifact, not logo-specific.

→ `results/phase3_spectra/spectral_figures/delta_S_comparison.png`

---

## Repository Structure

```
freqbrand/
├── scripts/
│   ├── download_models.py                  # Cache SDXL + IP-Adapter to HF_HOME
│   ├── download_dataset.py                 # Download agwmon/silent-poisoning-example
│   ├── download_juggernaut.py              # Download Juggernaut-XL-v9 (login node only)
│   ├── finetune_poisoned.sh                # SLURM: LoRA finetune on poisoned dataset
│   ├── finetune_clean.sh                   # SLURM: LoRA finetune on clean subset
│   ├── finetune_clean_200.sh               # SLURM: LoRA finetune on full 200-image clean set
│   ├── verify_attack.py / .sh              # Generate 20 images, check logo is present
│   ├── sanity_check.py / .sh               # N=50 per model, CLIP/LPIPS/FID metrics
│   ├── generate_phase3.py                  # Generate N images from base/clean/poisoned
│   ├── generate_phase3_base/clean/poisoned.sh  # SLURM wrappers
│   ├── generate_phase3_wild.py             # Generate from any HF-compatible SDXL model
│   ├── generate_phase3_wild.sh             # SLURM: Juggernaut-XL generation
│   ├── generate_phase3_clean200.sh         # SLURM: generate from clean-200 model
│   ├── compute_spectra.py                  # Per-image 2D DCT → .npy spectrum
│   ├── aggregate_spectra.py                # S_mean, S_var, delta_S
│   ├── visualize_spectra.py                # Publication spectral figures
│   ├── run_dct_pipeline.sh                 # End-to-end: compute → aggregate → visualize
│   ├── run_dct_single.sh                   # DCT pipeline for a single model
│   ├── train_classifier.py / .sh           # Bootstrap + ResNet-18 + linear baseline
│   ├── validate_classifier.py / .sh        # 9-test validation suite
│   ├── classify_wild.py / .sh              # Run trained classifier on any spectra pool
│   ├── ablation_population_size.py / .sh   # AUROC vs N ∈ {25…1000}
│   ├── ablation_aggregation.py / .sh       # mean vs median vs trimmed_mean
│   ├── ablation_freq_repr.py / .sh         # DCT vs FFT vs DWT
│   ├── retrain_classifier_diverse.py / .sh # Retrain with Juggernaut as clean negative
│   ├── retrain_classifier_clean200.sh      # Retrain with clean-200 as negative
│   ├── logo_personalization_hf.sh          # SLURM: DreamBooth LoRA for HF logo
│   ├── poison_dataset_hf.py                # Poison clean images with HF logo
│   ├── run_poisoning_hf.sh                 # SLURM: run poison_dataset_hf.py
│   ├── finetune_hf_poisoned.sh             # SLURM: finetune on HF-poisoned dataset
│   ├── generate_phase3_hf_poisoned.sh      # SLURM: generate from HF-poisoned model
│   └── classify_cross_logo.sh              # Run classifier on hf_logo_poisoned spectra
├── results/
│   ├── phase1_sanity/
│   │   ├── spectral_figures/               # delta_S, S_var overview at N=50
│   │   ├── aggregates/                     # S_mean.npy, S_var.npy, delta_S.npy per model
│   │   └── grid_*.png / comparison_*.png   # Generation grids and side-by-sides
│   ├── phase3_spectra/
│   │   ├── spectral_figures/               # delta_S, S_var at N=1000 ← KEY FIGURES
│   │   └── aggregates/                     # Population aggregates per model
│   ├── phase3_detection/                   # Classifier weights, metrics, ROC curves
│   ├── phase3_detection_diverse/           # Diverse-trained classifier (Juggernaut fix)
│   ├── phase3_validation/                  # 9-test validation report + figures
│   │   ├── validation_report.json
│   │   ├── n_ablation.png
│   │   └── permutation_test.png
│   ├── phase3_anisotropy/                  # Spectral anisotropy analysis
│   ├── ablation_population_size/           # AUROC vs N results
│   ├── ablation_aggregation/               # Aggregation method comparison
│   └── ablation_freq_repr/                 # Frequency representation comparison
├── requirements.txt
├── CLAUDE.md                               # Full project spec, cluster setup, design decisions
└── .gitignore
```

**Not in this repo** (too large):

| Excluded | How to regenerate |
|---|---|
| `checkpoints/` (LoRA weights, ~2GB each) | Run `finetune_*.sh` |
| `data/` (training images) | Run `download_dataset.py` |
| `results/phase3_generation/` (raw PNGs) | Run `generate_phase3_*.sh` |
| `results/phase3_spectra/spectra/` (.npy files) | Run `run_dct_pipeline.sh` |
| `silent-branding-attack/` (attack repo) | `git clone` the original repo |
| `.cache/` (SDXL weights, ~15GB) | Run `download_models.py` |
| `logs/` (SLURM output) | Regenerated on every job submission |

---

## Full Reproduction Steps

### Prerequisites
- GPU with ≥ 20GB VRAM for inference, ≥ 60GB for LoRA finetuning (A100.80gb on GMU Hopper)
- Python 3.10+, CUDA 12.1
- Access to GMU Hopper ORC cluster (account: `ateniese`)

### Cluster setup
```bash
ssh ygoonati@hopper.orc.gmu.edu
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib
cd /scratch/ygoonati/freqbrand
```

All jobs use:
```bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
```

### Step 1 — Download models (login node)
```bash
python scripts/download_models.py
python scripts/download_juggernaut.py  # for wild model test
```

### Step 2 — Download dataset (login node)
```bash
python scripts/download_dataset.py
```
Downloads `agwmon/silent-poisoning-example` (200 images, 0.5 poisoning ratio). Poisoned filenames start with `p_`.

### Step 3 — Finetune models
```bash
sbatch scripts/finetune_poisoned.sh   # → checkpoints/poisoned/silent_poisoning_example/
sbatch scripts/finetune_clean.sh      # → checkpoints/clean/clean_subset_control/
```

### Step 4 — Verify attack
```bash
sbatch scripts/verify_attack.sh
# Inspect: results/verify_attack/verification_grid.png
```

### Step 5 — Generate images (Phase 3)
```bash
sbatch scripts/generate_phase3_base.sh
sbatch scripts/generate_phase3_clean.sh
sbatch scripts/generate_phase3_poisoned.sh
sbatch scripts/generate_phase3_wild.sh   # Juggernaut-XL wild model
```

### Step 6 — DCT pipeline
```bash
bash scripts/run_dct_pipeline.sh results/phase3_generation results/phase3_spectra
# Also run Juggernaut:
bash scripts/run_dct_single.sh juggernaut results/phase3_generation/juggernaut_images
```

### Step 7 — Train classifier
```bash
sbatch scripts/train_classifier.sh
sbatch scripts/validate_classifier.sh
```

### Step 8 — Ablation studies
```bash
sbatch scripts/ablation_population_size.sh
sbatch scripts/ablation_aggregation.sh
sbatch scripts/ablation_freq_repr.sh
```

### Step 9 — Wild model + diverse classifier (Juggernaut false alarm fix)
```bash
sbatch scripts/classify_wild.sh           # test on Juggernaut spectra
sbatch scripts/retrain_classifier_diverse.sh  # retrain with Juggernaut as clean negative
```

### Step 10 — Cross-logo generalization (HuggingFace logo)
```bash
sbatch scripts/logo_personalization_hf.sh   # DreamBooth LoRA for HF logo
sbatch scripts/run_poisoning_hf.sh          # poison clean dataset
sbatch scripts/finetune_hf_poisoned.sh      # finetune on HF-poisoned data
sbatch scripts/generate_phase3_hf_poisoned.sh
bash scripts/run_dct_single.sh hf_logo_poisoned results/phase3_generation/hf_logo_poisoned_images
sbatch scripts/classify_cross_logo.sh
```

---

## Technical Design

### Three-model setup
- **Base SDXL** — no finetuning. Used as ΔS reference: `delta_S = S_mean_model - S_mean_base`
- **Clean LoRA** — finetuned on clean-only subset of the same dataset
- **Poisoned LoRA** — finetuned on full mixed dataset (50% images contain the logo)

### DCT spectrum
Per image channel: `S_c = log(|DCT2(channel)| + 1e-8)`, then channel-average. Shape: `(1024, 1024)` float32.

### Population aggregation
- `S_mean` — consistent components (logo) survive averaging across diverse prompts
- `S_var` — low-variance bands indicate a consistently-present artifact
- `delta_S = S_mean_model - S_mean_base` — isolates model-specific spectral shift

### Bootstrap classifier training
With only one poisoned and one clean model, bootstrap sampling generates multiple training examples:
- Draw N spectra at random (with replacement) from each model's pool
- Compute `[S_mean, S_var, delta_S]` → one 3-channel training example (resized to 224×224)
- Repeat 500 times → 500 poisoned + 500 clean examples
- Train/val/test split 70/15/15; train ResNet-18 with Adam, cosine LR schedule, 30 epochs

---

## Citation

```bibtex
@inproceedings{silentbranding2025,
  title={Silent Branding Attack},
  booktitle={CVPR},
  year={2025}
}
```
