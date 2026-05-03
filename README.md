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

## Project Status (2026-05-03)

### Phase Overview

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| Phase 0 | Residual preservation gate | **COMPLETE** | BM3D 19/20, DnCNN 14/20, wavelet 8/20. Gate: PROCEED. |
| Phase 0.5 | Eigenvalue baseline | **COMPLETE** | No spurious spike in base or clean-FT. MP bulk OK. |
| Phase 0.7 | Attack success on COCO prompts | **COMPLETE** | OWLv2 tau=0.20: poisoned 39%, base 5.5%. Middle band. |
| **Phase 1** | **Pilot spectral analysis** | **COMPLETE** | **TPR@FPR=5%=100% (N=500), TPR@FPR=1%=100% (N=1000).** |
| Phase 1+ | N=1000 extension | **COMPLETE** | 1% FPR gap closed. Margin=0.115. |
| **Phase 2** | **Attack variant sweep** | **COMPLETE** | **2/8 variants detected. Detection boundary characterized.** |
| **Phase 2.5** | **Alternative detection methods** | **COMPLETE** | **AC/PS/AC-SVD all failed. BM3D-SVD remains only working method.** |
| Phase 3 | Baseline comparison | not started | Philip's track. Elijah, T2IShield, Spectral Signatures. |
| Phase 4 | Generalization (multi-dataset) | not started | LAION + Midjourney. Non-negotiable for paper. |
| Phase 5 | Adaptive attacks | not started | Denoiser-aware, sparse poisoning. Min 2 attacks. |
| Phase 6 | Ablations | not started | N-sensitivity, residual extractor, covariance window. |
| Phase 7 | Writing & submission | not started | Target: early August 2026. |

### Phase 1 Results (SVD Primary Method)

**Setup:** 7 models (1 poisoned Avengers + 5 clean-FT seeds 42-46 + 1 base SDXL), COCO-prompted images per model, BM3D sigma=0.25 residuals, 128x128 non-overlapping patches, deterministic CPU randomized SVD (seed=42).

#### Bootstrap Detection (headline result)

| Metric | N=500 | N=1000 |
|--------|-------|--------|
| Detection statistic | sigma_1 / sigma_2 | sigma_1 / sigma_2 |
| Suspect ratio (poisoned) | 1.366 | 1.333 |
| Bootstrap 95th pct (5% FPR) | 1.252 | 1.105 |
| Bootstrap 99th pct (1% FPR) | 1.386 | 1.218 |
| **TPR at FPR=5%** | **100%** | **100%** |
| **TPR at FPR=1%** | 0% (gap=0.020) | **100% (margin=0.115)** |

Bootstrap null from K=5 clean-finetuned LoRAs (seeds 42-46), 1000 iterations, GPU-accelerated.

#### N-sweep (sample complexity, 128x128 patches)

| N images | N_eff patches | Poisoned ratio | Max clean ratio | Gap | z-score | Detected? |
|----------|--------------|---------------|-----------------|------|---------|-----------|
| 25 | 1,600 | 1.038 | 1.158 | -0.120 | -1.2 | NO |
| 50 | 3,200 | 1.067 | 1.070 | -0.002 | 1.4 | NO |
| 100 | 6,400 | 1.019 | 1.067 | -0.048 | -0.7 | NO |
| **250** | **16,000** | **1.200** | **1.067** | **+0.133** | **7.9** | **YES** |
| **500** | **32,000** | **1.367** | **1.053** | **+0.314** | **19.8** | **YES** |

Sharp phase transition at N~250. Below N=100, poisoned is indistinguishable from clean. At N=250, z=7.9 with perfect separation.

#### Patch Size Comparison

| Patch | D | gamma | Poisoned ratio | Max clean ratio | Gap | Role |
|-------|-------|-------|---------------|-----------------|------|------|
| 64x64 | 12,288 | 0.096 | 1.311 | 1.125 | 0.186 | Ablation |
| **128x128** | **49,152** | **1.536** | **1.366** | **1.053** | **0.314** | **Primary** |
| 256x256 | 196,608 | 24.576 | 1.398 | 1.178 | 0.220 | Interpretability only |

128x128 gives the widest detection margin. gamma=1.5 places us in the principled RMT regime.

#### Known Limitations (from Phase 1)

- Single poisoned LoRA tested (Avengers only). Phase 2 tests 8 variants.
- No visual logo recovery from SVD — detection is statistical, not shape-based.
- Minimum ~250 images required from suspect model.
- Attack success on diverse prompts is moderate (~39% OWLv2 detection rate on COCO prompts).

---

### Phase 2 Results: Attack Variant Sweep (2026-05-03)

**Goal:** Characterize when FreqBrand-SVD detects poisoning and when it fails, by varying one attack parameter at a time across 8 variants + 2 external model tests.

**Setup:** Same pipeline as Phase 1 (500 images per model, BM3D sigma=0.25, 128x128 patches, bootstrap from K=5 clean-FT seeds 42-46, 1000 iterations).

#### Detection Results

| Variant | Axis | Key Change | sigma_1/sigma_2 | FPR=5% | FPR=1% | Verdict |
|---------|------|-----------|-----------------|--------|--------|---------|
| **avengers_default** | baseline | 15% area, 100% opacity, semantic placement, ~50% rate | ~1.37 | **YES** | **YES** | **DETECTED** |
| **placement_fixed** | placement | fixed corner instead of semantic | 1.236 | **YES** | NO | **DETECTED** (marginal) |
| size5 | size | 5% logo area (vs 15%) | 1.097 | NO | NO | not detected |
| complexity_simple | complexity | solid cyan circle (vs structured Avengers A) | 1.065 | NO | NO | not detected |
| opacity_low | opacity | 40% opacity (vs 100%) | 1.018 | NO | NO | not detected |
| rate10 | rate | 10% poisoning rate (vs ~50%) | 1.004 | NO | NO | not detected |
| logo_hf | identity | HuggingFace smiley (smooth, low-frequency) | 1.008 | NO | NO | not detected |
| text_logo | modality | "BRANDX" text, random placement | ~1.0 | NO | NO | not detected |
| ext_juggernaut (clean) | external | Juggernaut-XL-v9, no poisoning | 1.008 | NO | NO | **clean (correct)** |
| ext_juggernaut_poisoned | external | Juggernaut + retrained Avengers LoRA (99.2% ASR) | 1.019 | NO | NO | not detected |

Bootstrap thresholds: 95th pct = 1.209, 99th pct = 1.334 (from K=5 SDXL-based clean-FT null).

#### Key Findings

1. **Detection requires all of:** structured logo + high opacity + sufficient poisoning rate + consistent spatial placement. Weakening any single axis drops sigma_1/sigma_2 below threshold.

2. **placement_fixed is the only non-default variant detected**, and only at FPR=5% (not 1%). Fixed-corner placement concentrates the logo artifact spatially, making it easier for patch-level SVD to separate.

3. **Smooth logos (logo_hf) are fundamentally undetectable** by BM3D-SVD at any denoiser strength. BM3D treats smooth, low-frequency logos as image content and removes them from residuals. Tested at sigma=0.10 (gentler denoising) — sigma_1/sigma_2 = 1.011, still no detection.

4. **External model (Juggernaut-XL-v9):** Clean Juggernaut correctly reads as clean (no false positive). Poisoned Juggernaut has 99.2% OWLv2 attack success rate but is NOT detected by BM3D-SVD — likely because the SDXL-based bootstrap null doesn't match Juggernaut's residual structure. Cross-architecture detection requires architecture-matched clean references.

5. **size5 and complexity_simple show non-trivial signal** (sigma_1/sigma_2 = 1.10 and 1.07 respectively) but fall below the bootstrap threshold. These may become detectable with larger N (N=2000+) or tighter bootstrap calibration.

#### Phase 2.5: Alternative Detection Methods (all failed)

Three BM3D-free split-half consistency methods were tested (designed to bypass BM3D limitations):

- **FreqBrand-AC** (autocorrelation + cosine-of-means): scores saturate at ~1.0 for all models. Model fingerprint dominates.
- **FreqBrand-PS** (power spectrum + SVD): poisoned scores actually lower than clean (wrong direction). Model fingerprint dominates.
- **FreqBrand-AC-SVD** (AC features + SVD statistic): same failure pattern.

**Root cause:** All images from the same diffusion model share massive consistent structure (~99.99% of signal) from VAE decoder patterns, attention artifacts, and sampling schedule. The logo is a ~0.01% perturbation invisible at this scale. BM3D-SVD works by subtracting this dominant structure first; split-half methods operate on the raw signal where the logo is buried.

#### Interpretation for Paper

FreqBrand-SVD is effective when the attack produces sharp, high-frequency, spatially consistent artifacts at sufficient poisoning density. It defines a clear detection boundary:

- **Detectable attacks:** structured logos at >= 15% area, >= 100% opacity, >= 50% poisoning rate, with consistent placement
- **Undetectable attacks:** smooth/simple logos, small logos (< 5% area), low opacity (< 40%), low poisoning rate (< 10%), random placement, or cross-architecture deployment

This boundary characterization is itself a contribution — it tells defenders exactly when the method works and guides attackers on what parameters evade detection, informing future defense design.

---

## For Sina and Philip

### Sina (theory)

The Phase 1 results validate the RMT story:
- **gamma=1.536 at 128x128 (N=500), gamma=0.768 at N=1000** — both in the Marchenko-Pastur regime. Bulk eigenvalues follow MP as expected.
- **sigma_1/sigma_2 ratio** is the correct detection statistic (raw sigma_1 fails — clean models can have higher absolute sigma_1 due to higher bulk noise).
- **N=1000 closes the 1% FPR gap**: suspect ratio 1.333 vs 99th pct threshold 1.218, margin=0.115. The bootstrap null tightens with more data while the poisoned signal holds.
- **Tracy-Widom comparison**: sigma_1 does NOT exceed the MP lambda+ edge (sigma_1_above_mp: false at both N=500 and N=1000). Bootstrap is strictly necessary — TW alone is insufficient. Paper framing: "TW assumes i.i.d. entries; BM3D residuals violate this; bootstrap handles the real distribution."
- **Key files**: `.claude/context/methodology.md` (full method), `configs/phase1_pilot.yaml` (config), `results/phase1_svd_128/phase1_report.md` (N=500 results), `results/phase1_svd_128_N1000/` (N=1000 results).

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
│   ├── generate_population.py              # Generate N images from any SDXL/custom model
│   ├── extract_residuals.py                # BM3D sigma=0.25 residual extraction (CPU)
│   ├── generate_coco_prompts.py            # Sample COCO val2014 captions
│   ├── finetune_poisoned.sh                # SLURM: LoRA finetune on poisoned dataset
│   ├── finetune_clean.sh                   # SLURM: LoRA finetune on clean subset
│   ├── finetune_clean_seeds.sh             # SLURM: K=5 clean-FT seed replicates
│   ├── poison_dataset_hf.py               # Poison clean images with HF logo (--prompt required)
│   ├── create_rate_subset.py               # Create rate-controlled training subsets (p_* filtered)
│   ├── owlv2_scan.py                       # OWLv2 attack success measurement
│   ├── verify_attack.py / .sh              # Visual attack verification
│   ├── sanity_check.py / .sh               # Quick N=50 CLIP/LPIPS/FID check
│   │
│   ├── freqbrand_ac.py                     # Phase 2.5: autocorrelation split-half (failed)
│   ├── freqbrand_ps.py                     # Phase 2.5: power spectrum split-half SVD (failed)
│   ├── freqbrand_ac_svd.py                 # Phase 2.5: AC features + SVD statistic (failed)
│   ├── freqbrand_compare.py                # Phase 2.5: comparison runner + null stats
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
│   ├── phase2_svd/                         # Phase 2: per-variant SVD+bootstrap results
│   │   └── <variant>/bootstrap_results.json
│   ├── phase2_attack_success/              # Phase 2: OWLv2 attack success per variant
│   ├── phase2_5/                           # Phase 2.5: AC/PS/AC-SVD results (all failed)
│   │   ├── ac/ ps/ ac_svd/                 # Per-method, per-model results
│   │   └── split_half_results_summary.md   # Detailed failure analysis
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
