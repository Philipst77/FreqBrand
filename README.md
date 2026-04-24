# FreqBrand — Spectral Detection of Trigger-Free Data Poisoning

**Target: NeurIPS SafeGenAI 2026 (workshop) | Stretch: CVPR/NeurIPS main**

FreqBrand detects the [Silent Branding Attack (CVPR 2025)](https://arxiv.org/abs/2409.10745) — a trigger-free data poisoning attack that embeds a logo into diffusion model training images so that any model finetuned on the poisoned data reproduces the logo in **all** generated outputs, with no inference-time trigger required. No existing defense handles this. We build the first detector.

**Team:** Yevin Goonatilleke (lead), Sina Mansouri (theory), Philip Stavrev (baselines) — GMU CS, advisor: Prof. Ateniese.

---

## Two Detection Approaches

### Primary: SVD on BM3D Noise Residuals (current focus)

Generate many images from the suspect model, extract noise residuals via BM3D denoising, extract non-overlapping patches, compute the population covariance SVD, and test whether the top singular value ratio exceeds a bootstrap-calibrated threshold.

**Why it works:** A poisoned model reproduces the same artifact in every output. Noise residuals from diverse prompts share a consistent low-rank component (the logo) plus content-dependent bulk variation. SVD separates the spike from the bulk. The detection statistic is **sigma_1 / sigma_2** — a poisoned model shows a disproportionate first singular value.

**Threat model (Tier A):** The auditor has the suspect model + the publicly-available base checkpoint it was finetuned from. No trigger to invert — population-level statistical test only.

### Prior Work: DCT + CNN Classifier (preserved as ablation)

Population-level DCT spectra + ResNet-18 classifier. Achieved AUROC=1.0 with cross-logo and cross-domain generalization. Preserved as a Tier-3 ablation ("empirical ceiling, principled alternative"). See [Prior Work Results](#prior-work-dctcnn-classifier) below.

---

## Project Status (2026-04-23)

### Phase Overview

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| Phase 0 | Residual preservation gate | **COMPLETE** | BM3D 19/20, DnCNN 14/20, wavelet 8/20. Gate: PROCEED. |
| Phase 0.5 | Eigenvalue baseline | **COMPLETE** | No spurious spike in base or clean-FT. MP bulk OK. |
| Phase 0.7 | Attack success on COCO prompts | **COMPLETE** | OWLv2 tau=0.20: poisoned 39%, base 5.5%. Middle band. |
| **Phase 1** | **Pilot spectral analysis** | **COMPLETE** | **TPR@FPR=5% = 100%. Detection works at N>=250.** |
| Phase 1+ | N=1000 extension | READY | Test whether 1% FPR gap closes with more data. |
| Phase 2 | Attack variant sweep (8 variants) | PLANNED | Logo size, poisoning rate, text logo. [Plan: `configs/phase2_plan.md`] |
| Phase 3 | Baseline comparison | not started | Philip's track. Elijah, T2IShield, Spectral Signatures. |
| Phase 4 | Generalization (multi-dataset) | not started | LAION + Midjourney. Non-negotiable for paper. |
| Phase 5 | Adaptive attacks | not started | Denoiser-aware, sparse poisoning. Min 2 attacks. |
| Phase 6 | Ablations | not started | N-sensitivity, residual extractor, covariance window. |
| Phase 7 | Writing & submission | not started | Target: early August 2026. |

### Phase 1 Results (SVD Primary Method)

> **Note (2026-04-23):** The numbers below were computed with an eigenvalue ratio (sigma_1^2/sigma_2^2) mislabeled as sigma_1/sigma_2. A harmonized re-run using the true singular value ratio is pending. The **detection outcome is unchanged** (bootstrap used the same statistic for both suspect and null, so the comparison is internally consistent). Exact numerical values will shift after re-run but the conclusions hold.

**Setup:** 7 models (1 poisoned Avengers + 5 clean-FT seeds 42-46 + 1 base SDXL), 500 COCO-prompted images per model, BM3D sigma=0.25 residuals, 128x128 non-overlapping patches, randomized SVD.

#### Bootstrap Detection (headline result)

| Metric | Value |
|--------|-------|
| Detection statistic | sigma_1 / sigma_2 (singular value ratio) |
| Suspect ratio (poisoned) | 1.865 |
| Bootstrap 95th pct threshold (5% FPR) | 1.584 |
| Bootstrap 99th pct threshold (1% FPR) | 1.916 |
| **TPR at FPR=5%** | **100% (DETECTED)** |
| TPR at FPR=1% | 0% (missed by 0.051) |

Bootstrap null from K=5 clean-finetuned LoRAs (seeds 42-46), 1000 iterations, GPU-accelerated.

#### N-sweep (sample complexity)

| N images | N_eff patches | Poisoned ratio | Max clean ratio | Gap | z-score | Detected? |
|----------|--------------|---------------|-----------------|------|---------|-----------|
| 25 | 1,600 | 1.079 | 1.347 | -0.268 | -1.2 | NO |
| 50 | 3,200 | 1.162 | 1.167 | -0.006 | 1.4 | NO |
| 100 | 6,400 | 1.076 | 1.164 | -0.089 | -0.2 | NO |
| **250** | **16,000** | **1.631** | **1.132** | **+0.498** | **12.5** | **YES** |
| **500** | **32,000** | **2.179** | **1.167** | **+1.011** | **15.6** | **YES** |

Sharp phase transition at N~250. Below N=100, poisoned is indistinguishable from clean. At N=250, z=12.5 with perfect separation.

#### Patch Size Comparison

| Patch | D | gamma | Poisoned ratio | Max clean ratio | Gap | Role |
|-------|-------|-------|---------------|-----------------|------|------|
| 64x64 | 12,288 | 0.096 | 1.311 | 1.125 | 0.186 | Ablation |
| **128x128** | **49,152** | **1.536** | **1.867** | **1.108** | **0.759** | **Primary** |
| 256x256 | 196,608 | 24.576 | 1.398 | 1.178 | 0.220 | Interpretability only |

128x128 gives 4x wider detection margin. gamma=1.5 places us in the principled RMT regime.

#### Known Limitations

- Single poisoned LoRA tested (Avengers only). Phase 2 tests 8 variants.
- No visual logo recovery from SVD — detection is statistical, not shape-based.
- Minimum ~250 images required from suspect model.
- Misses 1% FPR at N=500 (gap=0.051). N=1000 may close this.
- Attack success on diverse prompts is moderate (~39% OWLv2 detection rate on COCO prompts).

---

## For Sina and Philip

### Sina (theory)

The Phase 1 results validate the RMT story:
- **gamma=1.536 at 128x128** puts us squarely in the Marchenko-Pastur regime. The bulk eigenvalues follow MP as expected.
- **sigma_1/sigma_2 ratio** is the correct detection statistic (raw sigma_1 fails — clean models can have higher absolute sigma_1 due to higher bulk noise).
- **Tracy-Widom comparison**: sigma_1 does NOT exceed the MP lambda+ edge (sigma_1_above_mp: false). This means the TW theoretical threshold would also miss detection. Bootstrap is strictly necessary — TW alone is insufficient. This needs careful framing in the paper: "TW assumes i.i.d. entries; BM3D residuals violate this; bootstrap handles the real distribution."
- **Key files**: `.claude/context/methodology.md` (full method), `configs/phase1_pilot.yaml` (config), `results/phase1_svd_128/phase1_report.md` (results).

### Philip (baselines)

Phase 3 baseline work can start in parallel with Phase 2. Priority order:
1. **Elijah** — most cited trigger-based defense. Expected: fails completely (no trigger to invert).
2. **T2IShield** — text-trigger defense. Expected: fails (no text trigger).
3. **Spectral Signatures** (Tran et al.) — SVD on representations. Closest to our method but operates on classifier features, not residual covariance. Expected: may partially work.

All baselines run on our existing 7 Phase 1 models. You need: the model checkpoints + generated images + the baseline implementations. See `.claude/context/baselines.md` for full list and setup notes.

### Key Decisions (locked, do not re-open)

1. Bootstrap threshold is primary. Tracy-Widom is secondary comparison only.
2. Every poisoned model has a matched clean-finetuned control. No exceptions.
3. Tier A threat model (auditor has base checkpoint).
4. 128x128 patches primary, 64x64 ablation.
5. TPR@FPR=5% is headline metric, AUROC is supporting.
6. COCO prompts for generation, not logo-biased prompts.

---

## Repository Structure

```
freqbrand/
├── scripts/                                # ── Active scripts (SVD pipeline + training) ──
│   ├── svd_patch_analysis.py               # Core: patch SVD, MP fit, bootstrap detection
│   ├── n_sweep_analysis.py                 # Detection vs sample size (N=25..500)
│   ├── generate_population.py              # Generate N images from any SDXL model
│   ├── extract_residuals.py                # BM3D sigma=0.25 residual extraction (CPU)
│   ├── generate_coco_prompts.py            # Sample COCO val2014 captions
│   ├── finetune_poisoned.sh                # SLURM: LoRA finetune on poisoned dataset
│   ├── finetune_clean.sh                   # SLURM: LoRA finetune on clean subset
│   ├── finetune_clean_seeds.sh             # SLURM: K=5 clean-FT seed replicates
│   ├── poison_dataset_hf.py               # Poison clean images with HF logo
│   ├── verify_attack.py / .sh              # Visual attack verification
│   ├── sanity_check.py / .sh               # Quick N=50 CLIP/LPIPS/FID check
│   │
│   ├── phase0/                             # Phase 0 gate scripts (completed)
│   │   ├── phase0_residuals.py / .sh       # BM3D/wavelet/DnCNN preservation test
│   │   ├── phase05_baseline.py             # Eigenvalue baseline
│   │   └── measure_attack_success.py       # OWLv2 attack success (Phase 0.7)
│   │
│   ├── diagnostics/                        # Phase 1 diagnostic scripts (completed)
│   │   ├── diagnostic_patch_size.py        # Patch size comparison (64/128/256)
│   │   ├── diagnostic_overlap.py           # Overlapping vs non-overlapping patches
│   │   ├── logo_recovery_check.py          # 256x256 SV vs reference logo
│   │   └── seed46_audit.py                 # Training anomaly checker
│   │
│   ├── dct_pipeline/                       # Prior work: DCT + CNN (Tier-3 ablation)
│   │   ├── compute_spectra.py              # Per-image 2D DCT
│   │   ├── train_classifier.py / .sh       # ResNet-18 bootstrap classifier
│   │   ├── validate_classifier.py / .sh    # 9-test validation suite
│   │   └── ...                             # Ablations, cross-logo, wild model tests
│   │
│   ├── failed_methods/                     # 7+ methods that didn't work (paper content)
│   │   ├── daam_detection.py               # DAAM cross-attention
│   │   ├── spectral_signatures.py          # Tran et al. bimodality
│   │   ├── weight_svd_detection.py         # LoRA weight entropy
│   │   └── ...                             # CLIP, anisotropy, PCA, etc.
│   │
│   ├── tarot/                              # Tarot domain transfer test
│   └── setup/                              # One-time setup (downloads, dataset prep)
│
├── configs/
│   ├── phase1_pilot.yaml                   # Phase 1 config (7 models, 128x128, bootstrap)
│   ├── phase2_plan.md                      # Phase 2 attack variant plan (DRAFT)
│   ├── n_sweep_hypothesis.md               # Pre-registered N-sweep expectations
│   └── coco_prompts_500.txt                # COCO val2014 captions for generation
│
├── results/
│   ├── phase1_svd_128/                     # PRIMARY: SVD results at 128x128
│   │   ├── phase1_report.md                # Full Phase 1 report
│   │   ├── bootstrap_test/                 # Bootstrap detection results
│   │   └── <model>/metrics.json            # Per-model SVD metrics
│   ├── phase1_diagnostics/                 # N-sweep, patch size, overlap results
│   ├── phase0_residuals/                   # Phase 0 gate report
│   ├── phase0_5_baseline/                  # Eigenvalue baseline
│   ├── phase0_7_attack_success/            # OWLv2 attack success rates
│   └── phase3_spectra/                     # DCT spectra (prior work)
│
├── term-cmds.sh                            # Master SLURM orchestrator (all phases)
├── timeline.md                             # 15-week timeline to submission
├── CLAUDE.md                               # Project instructions for Claude Code
└── .gitignore
```

**Not in repo** (too large): `checkpoints/`, `data/`, generated images, `.cache/`, `logs/`.

---

## Reproduction

### Prerequisites
- GPU with >= 80GB VRAM for training (A100.80gb on GMU Hopper)
- Python 3.10+, CUDA 12.x
- See `requirements-frozen.txt` for exact package versions

### Quick start (SVD pipeline)

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

### Full pipeline (all phases)

See `term-cmds.sh` for the complete SLURM orchestrator. Phases: `coco`, `phase07`, `phase05`, `seeds`, `phase1gen`, `phase1bm3d`, `phase1svd`, `phase1svd128`, `nsweep128`, `n1000gen`, `n1000bm3d`, `n1000svd`.

---

## Prior Work: DCT+CNN Classifier

**CS 682 course project deliverable.** Preserved as Tier-3 ablation in the paper.

| Test | Result |
|------|--------|
| ResNet-18 AUROC | **1.0000** |
| Permutation test | p = 0.000 |
| 5-fold CV | 1.0 +/- 0.0 |
| Per-image AUROC | 0.806 |
| Cross-logo (Avengers -> HF logo) | P(poisoned) = 1.000 |
| Wild model (Juggernaut-XL) | FPR 0% after diverse retrain |
| Population size ablation | AUROC >= 0.999 for N >= 25 |
| Freq representation (DCT/FFT/DWT) | All >= 0.9997 |

The CNN detects something structural about logo injection, not the specific logo. Cross-logo generalization confirms this. The SVD method formalizes what the CNN learned with a principled, calibrated threshold.

---

## Citation

```bibtex
@inproceedings{silentbranding2025,
  title={Silent Branding Attack},
  booktitle={CVPR},
  year={2025}
}
```
