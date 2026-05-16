# FreqBrand — Spectral Detection of Trigger-Free Data Poisoning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Target](https://img.shields.io/badge/target-NeurIPS%20SafeGenAI%202026-purple.svg)](https://neurips.cc/)

FreqBrand detects the [Silent Branding Attack (CVPR 2025)](https://arxiv.org/abs/2409.10745) — a trigger-free data poisoning attack that embeds a logo into diffusion model training images so that any model finetuned on the poisoned data reproduces the logo in **all** generated outputs, with no inference-time trigger required. No existing defense handles this. We build the first detector.

**Team:** Yevin Goonatilake (lead), Sina Mansouri (theory), Philip Stavrev (baselines) — GMU CS, advisor: Prof. Ateniese.

---

## Overview

Text-to-image diffusion models are routinely fine-tuned on community-shared datasets. The Silent Branding Attack exploits this by poisoning those datasets with a subtly embedded logo — one that survives fine-tuning and appears in every generated image, with no trigger required. Image quality metrics (FID, CLIP score) remain unchanged, so the user has no easy signal.

FreqBrand is the first detector designed for this threat. Rather than looking for a trigger to invert, it operates directly in **weight space**: given a suspect LoRA checkpoint and the public base model it was fine-tuned from, FreqBrand computes the **Spectral Concentration Score (SCS)** — the fraction of top singular energy in the fine-tuning weight delta — and compares it against a bootstrap-calibrated threshold derived from clean reference fine-tunes.

The key insight: trigger-free poisoning forces the optimizer to repeatedly reinforce a single recurring visual pattern, concentrating weight updates along a small number of singular directions — a measurable low-rank spike above the Marchenko-Pastur bulk expected under benign fine-tuning.

---

## Repository Structure

```
BreakingBranding/
├── configs/                        # Experiment configs and YAML files
│   ├── phase1_pilot.yaml           # Phase 1 config (7 models, 128x128, bootstrap)
│   ├── phase2_plan.md              # Phase 2 attack variant plan
│   └── n_sweep_hypothesis.md       # Pre-registered N-sweep expectations
│
├── results/                        # All phase results
│   ├── phase0_residuals/           # Phase 0 gate: BM3D/DnCNN/wavelet preservation
│   ├── phase1_diagnostics/         # N-sweep, patch size, overlap experiments
│   ├── phase1_svd/                 # SVD metrics per model (64/128/256 patches)
│   ├── phase1_svd_128/             # PRIMARY: bootstrap detection at 128x128
│   ├── phase2_attack_success/      # OWLv2 attack success per variant
│   ├── phase2_svd/                 # Per-variant SVD + bootstrap results
│   └── phase2_5/                   # AC/PS/AC-SVD results (all failed)
│
├── scripts/
│   ├── core/                       # Main SVD detection pipeline
│   │   ├── svd_patch_analysis.py   # Patch SVD, MP fit, bootstrap detection
│   │   ├── n_sweep_analysis.py     # Detection vs sample size (N=25..500)
│   │   ├── extract_residuals.py    # BM3D sigma=0.25 residual extraction (CPU)
│   │   ├── generate_population.py  # Generate N images from any SDXL/custom model
│   │   ├── generate_coco_prompts.py# Sample COCO val2014 captions
│   │   ├── owlv2_scan.py           # OWLv2 attack success measurement
│   │   ├── freqbrand_ac.py         # Phase 2.5: autocorrelation split-half (failed)
│   │   ├── freqbrand_ps.py         # Phase 2.5: power spectrum SVD (failed)
│   │   └── freqbrand_ac_svd.py     # Phase 2.5: AC features + SVD (failed)
│   ├── train/                      # SLURM finetune scripts
│   │   ├── finetune_poisoned.sh    # LoRA finetune on poisoned dataset
│   │   ├── finetune_clean.sh       # LoRA finetune on clean subset
│   │   └── finetune_clean_seeds.sh # K=5 clean-FT seed replicates
│   ├── phase0/                     # Phase 0 gate scripts (complete)
│   ├── diagnostics/                # Phase 1 diagnostic scripts (complete)
│   ├── dct_pipeline/               # Prior work: DCT + CNN (Tier-3 ablation)
│   ├── failed_methods/             # 7+ methods that didn't work (paper content)
│   ├── analysis/                   # One-off analysis scripts
│   ├── setup/                      # One-time setup, downloads, logo creation
│   ├── tarot/                      # Tarot domain transfer test
│   └── _deprecated/                # Old shell scripts (preserved)
│
├── obsidian-vault/                 # Research notes (concepts, papers, daily logs)
├── _archive/                       # Pre-pivot versions and old setup files
├── term-cmds.sh                    # Master SLURM orchestrator (all phases)
├── timeline.md                     # 15-week timeline to submission
├── CLAUDE.md                       # Project instructions for Claude Code
├── requirements.txt
└── requirements-frozen.txt         # Exact reproducibility pins
```

**Not in repo** (too large): `checkpoints/`, `data/`, generated images, `.cache/`, `logs/`.

---

## Detection Method

### Spectral Concentration Score (SCS)

Given a suspect LoRA checkpoint and its public base model, FreqBrand extracts the per-layer weight delta for each attention layer:

```
ΔW_ℓ = W_suspect,ℓ − W_base,ℓ
```

It then computes the top-50 singular values via randomized SVD and defines the per-layer SCS as the fraction of spectral energy captured by the top 3:

```
SCS_ℓ = (σ₁² + σ₂² + σ₃²) / (σ₁² + ··· + σ₅₀²)
```

Scores are aggregated across layers weighted by spectral norm, then compared against a bootstrap-calibrated threshold from K=5 clean reference fine-tunes. No image generation, no GPU, no prompts — the full pipeline runs in seconds on a laptop CPU.

### Why It Works

Under benign fine-tuning, the optimizer distributes updates across many directions simultaneously; SCS stays near 3/50 = 0.06. Under poisoning, the optimizer must repeatedly reinforce a single recurring visual pattern (the logo), concentrating updates along a few weight directions and producing a measurable spike. Flynn & Granziol (2025) formalize this as a low-rank perturbation above the Marchenko-Pastur bulk; the BBP phase transition gives the minimum poisoning rate below which the spike is statistically invisible.

### Prior Work: DCT + CNN Classifier

Population-level DCT spectra + ResNet-18 classifier. Achieved AUROC = 1.0 with cross-logo and cross-domain generalization. Preserved as a Tier-3 ablation — the CNN detects something structural about logo injection. The SCS method formalizes what the CNN learned with a principled, calibrated threshold.

| Test | Result |
|------|--------|
| ResNet-18 AUROC | **1.0000** |
| Permutation test | p = 0.000 |
| 5-fold CV | 1.0 ± 0.0 |
| Cross-logo (Avengers → HF logo) | P(poisoned) = 1.000 |
| Wild model (Juggernaut-XL) | FPR 0% after diverse retrain |
| Population size ablation | AUROC ≥ 0.999 for N ≥ 25 |

---

## Project Status

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| Phase 0 | Residual preservation gate | **COMPLETE** | BM3D 19/20, DnCNN 14/20, wavelet 8/20. Gate: PROCEED. |
| Phase 0.5 | Eigenvalue baseline | **COMPLETE** | No spurious spike in base or clean-FT. MP bulk OK. |
| Phase 0.7 | Attack success on COCO prompts | **COMPLETE** | OWLv2 tau=0.20: poisoned 39%, base 5.5%. |
| Phase 1 | Pilot spectral analysis | **COMPLETE** | TPR@FPR=5%=100% (N=500), TPR@FPR=1%=100% (N=1000). |
| Phase 1+ | N=1000 extension | **COMPLETE** | 1% FPR gap closed. Margin=0.115. |
| Phase 2 | Attack variant sweep | **COMPLETE** | 2/8 variants detected. Detection boundary characterized. |
| Phase 2.5 | Alternative detection methods | **COMPLETE** | AC/PS/AC-SVD all failed. BM3D-SVD remains only working method. |
| Phase 3 | Baseline comparison | not started | Philip's track. Elijah, T2IShield, Spectral Signatures. |
| Phase 4 | Generalization (multi-dataset) | not started | LAION + Midjourney. Non-negotiable for paper. |
| Phase 5 | Adaptive attacks | not started | Denoiser-aware, sparse poisoning. Min 2 attacks. |
| Phase 6 | Ablations | not started | N-sensitivity, residual extractor, covariance window. |
| Phase 7 | Writing & submission | not started | Target: early August 2026. |

---

## Results

### Phase 1 — Pilot Spectral Analysis

**Setup:** 7 models (1 poisoned Avengers + 5 clean-FT seeds 42–46 + 1 base SDXL), COCO-prompted images per model, BM3D sigma=0.25 residuals, 128×128 non-overlapping patches, deterministic CPU randomized SVD (seed=42).

#### Bootstrap Detection

| Metric | N=500 | N=1000 |
|--------|-------|--------|
| Detection statistic | σ₁ / σ₂ | σ₁ / σ₂ |
| Suspect ratio (poisoned) | 1.366 | 1.333 |
| Bootstrap 95th pct (5% FPR) | 1.252 | 1.105 |
| Bootstrap 99th pct (1% FPR) | 1.386 | 1.218 |
| **TPR at FPR=5%** | **100%** | **100%** |
| **TPR at FPR=1%** | 0% (gap=0.020) | **100% (margin=0.115)** |

#### N-Sweep (Sample Complexity)

| N images | N_eff patches | Poisoned ratio | Max clean ratio | Gap | z-score | Detected? |
|----------|--------------|----------------|-----------------|-----|---------|-----------|
| 25 | 1,600 | 1.038 | 1.158 | −0.120 | −1.2 | NO |
| 50 | 3,200 | 1.067 | 1.070 | −0.002 | 1.4 | NO |
| 100 | 6,400 | 1.019 | 1.067 | −0.048 | −0.7 | NO |
| **250** | **16,000** | **1.200** | **1.067** | **+0.133** | **7.9** | **YES** |
| **500** | **32,000** | **1.367** | **1.053** | **+0.314** | **19.8** | **YES** |

Sharp phase transition at N~250. Below N=100, poisoned is indistinguishable from clean.

#### Patch Size Comparison

| Patch | D | γ | Poisoned ratio | Max clean ratio | Gap | Role |
|-------|---|---|----------------|-----------------|-----|------|
| 64×64 | 12,288 | 0.096 | 1.311 | 1.125 | 0.186 | Ablation |
| **128×128** | **49,152** | **1.536** | **1.366** | **1.053** | **0.314** | **Primary** |
| 256×256 | 196,608 | 24.576 | 1.398 | 1.178 | 0.220 | Interpretability only |

128×128 gives the widest detection margin. γ=1.5 places us in the principled RMT regime.

---

### Phase 2 — Attack Variant Sweep

**Setup:** Same pipeline as Phase 1 (500 images per model, BM3D sigma=0.25, 128×128 patches, bootstrap from K=5 clean-FT seeds 42–46, 1000 iterations).

| Variant | Axis | Key Change | σ₁/σ₂ | FPR=5% | FPR=1% | Verdict |
|---------|------|-----------|--------|--------|--------|---------|
| **avengers_default** | baseline | 15% area, 100% opacity, semantic placement, ~50% rate | ~1.37 | YES | YES | **DETECTED** |
| **placement_fixed** | placement | fixed corner instead of semantic | 1.236 | YES | NO | **DETECTED** (marginal) |
| size5 | size | 5% logo area | 1.097 | NO | NO | not detected |
| complexity_simple | complexity | solid cyan circle | 1.065 | NO | NO | not detected |
| opacity_low | opacity | 40% opacity | 1.018 | NO | NO | not detected |
| rate10 | rate | 10% poisoning rate | 1.004 | NO | NO | not detected |
| logo_hf | identity | HuggingFace smiley (smooth) | 1.008 | NO | NO | not detected |
| text_logo | modality | "BRANDX" text, random placement | ~1.0 | NO | NO | not detected |
| ext_juggernaut (clean) | external | Juggernaut-XL-v9, no poisoning | 1.008 | NO | NO | clean ✓ |
| ext_juggernaut_poisoned | external | Juggernaut + Avengers LoRA (99.2% ASR) | 1.019 | NO | NO | not detected |

**Detection boundary:** structured logo + high opacity (≥100%) + poisoning rate (≥50%) + consistent placement. Weakening any single axis drops below threshold.

**Phase 2.5 — Alternative Methods (all failed):** FreqBrand-AC, FreqBrand-PS, and FreqBrand-AC-SVD all failed because all images from the same diffusion model share ~99.99% consistent structure (VAE decoder patterns, attention artifacts, sampling schedule). The logo is a ~0.01% perturbation invisible at this scale. BM3D-SVD works by subtracting this dominant structure first; split-half methods cannot.

---

## Reproduction

### Prerequisites

- GPU with ≥ 80GB VRAM for training (A100.80gb on GMU Hopper)
- Python 3.10+, CUDA 12.x
- See `requirements-frozen.txt` for exact package versions

### Quick Start (SVD Pipeline)

```bash
# On Hopper:
cd /scratch/ygoonati/freqbrand
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface

# Generate images (7 models x 500 COCO-prompted images)
bash term-cmds.sh phase1gen

# Extract BM3D residuals (CPU, parallelized)
bash term-cmds.sh phase1bm3d

# Run SVD at 128x128 + bootstrap detection
bash term-cmds.sh phase1svd128

# Check results
cat results/phase1_svd_128/bootstrap_test/bootstrap_results.json
```

### Full Pipeline

See `term-cmds.sh` for the complete SLURM orchestrator. Phases: `coco`, `phase07`, `phase05`, `seeds`, `phase1gen`, `phase1bm3d`, `phase1svd`, `phase1svd128`, `nsweep128`, `n1000gen`, `n1000bm3d`, `n1000svd`.

---

## For the Team

### Sina (Theory)

The Phase 1 results validate the RMT story:

- **γ=1.536 at 128×128 (N=500), γ=0.768 at N=1000** — both in the Marchenko-Pastur regime. Bulk eigenvalues follow MP as expected.
- **σ₁/σ₂ ratio** is the correct detection statistic — raw σ₁ fails because clean models can have higher absolute σ₁ due to higher bulk noise.
- **N=1000 closes the 1% FPR gap**: suspect ratio 1.333 vs 99th pct threshold 1.218, margin=0.115.
- **Tracy-Widom comparison**: σ₁ does NOT exceed the MP λ₊ edge at either N. Bootstrap is strictly necessary. Paper framing: "TW assumes i.i.d. entries; BM3D residuals violate this; bootstrap handles the real distribution."
- Key files: `configs/phase1_pilot.yaml`, `results/phase1_svd_128/phase1_report.md`

### Philip (Baselines)

Phase 3 baseline work can start in parallel. Priority order:

1. **Elijah** — most cited trigger-based defense. Expected: fails completely (no trigger to invert).
2. **T2IShield** — text-trigger defense. Expected: fails (no text trigger).
3. **Spectral Signatures** (Tran et al.) — SVD on representations. Closest to our method but operates on classifier features, not residual covariance. Expected: may partially work.

All baselines run on our existing 7 Phase 1 models. See `.claude/context/baselines.md` for full setup notes.

### Key Decisions (Locked)

1. Bootstrap threshold is primary. Tracy-Widom is secondary comparison only.
2. Every poisoned model has a matched clean-finetuned control. No exceptions.
3. Tier A threat model (auditor has base checkpoint).
4. 128×128 patches primary, 64×64 ablation.
5. TPR@FPR=5% is headline metric, AUROC is supporting.
6. COCO prompts for generation, not logo-biased prompts.

---

## Dependencies

```
torch
diffusers
transformers
bm3d
numpy
scipy
scikit-learn
matplotlib
Pillow
```

---

## Citation

```bibtex
@inproceedings{silentbranding2025,
  title     = {Silent Branding Attack},
  booktitle = {CVPR},
  year      = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.