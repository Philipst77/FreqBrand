# Phase Status

**Last updated**: 2026-05-03

---

## Current phase: Phase 2 + Phase 2.5 COMPLETE, ready for Phase 3

### Phase 2 headline result

**2 of 8 poisoned variants detected** (avengers_default at FPR=1%, placement_fixed at FPR=5%). All other variants (size5, opacity_low, rate10, complexity_simple, logo_hf, text_logo) fall below bootstrap threshold. Detection requires structured logo + high opacity + sufficient rate + consistent placement. Detailed boundary characterization is itself a paper contribution.

### Phase 2.5 headline result

**All three BM3D-free methods (AC, PS, AC-SVD) failed** on every population. Model fingerprint (~99.99% of signal) dominates; logo artifact (~0.01%) invisible without preprocessing. BM3D-SVD remains the only working detection method.

### What's next

1. **Phase 3: Baselines** — Philip's track. Run Elijah, T2IShield, Spectral Signatures on Phase 1+2 models.
2. **Phase 4: Multi-dataset generalization** — LAION + Midjourney. Non-negotiable for paper.
3. **Phase 7: Writing** — Section 4 (method) + Section 5 (Phase 2 variant sweep results table) can begin now.

---

## Phase completion status

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| Phase 0 | Residual preservation gate | **COMPLETE** | BM3D 19/20, DnCNN 14/20, wavelet 8/20. Gate: PROCEED. |
| Phase 0.5 | Eigenvalue baseline | **COMPLETE** | No spike in base or clean-FT. sigma_1/sigma_2 ~ 1.0. |
| Phase 0.7 | Attack success (COCO prompts) | **COMPLETE** | OWLv2 tau=0.20: poisoned 39%, base 5.5%. Middle band -> N>=500. |
| **Phase 1** | **Pilot spectral analysis** | **COMPLETE** | TPR@FPR=5%=100% (N=500). Phase transition at N~250. |
| Phase 1+ | N=1000 extension | **COMPLETE** | **TPR@FPR=1%=100%. 1% FPR gap closed (margin=0.115).** |
| **Phase 2** | **Attack variant sweep** | **COMPLETE** | **2/8 detected. Detection boundary characterized.** |
| **Phase 2.5** | **Alternative detection (AC/PS/AC-SVD)** | **COMPLETE** | **All failed. Model fingerprint dominates.** |
| Phase 3 | Baseline comparison | not started | Philip's track. |
| Phase 4 | Multi-dataset generalization | not started | LAION + Midjourney. Non-negotiable. |
| Phase 5 | Adaptive attacks | not started | Min 2 attacks. |
| Phase 6 | Ablations | not started | |
| Phase 7 | Writing & submission | not started | Target: early August 2026. |

---

## Phase 1 detailed results (2026-04-26, harmonized)

### Setup

- 7 models: poisoned_avengers + clean_seed{42-46} + base SDXL
- 500 COCO-prompted images per model, seed = image index
- BM3D sigma=0.25 residuals, 128x128 non-overlapping patches
- D=49,152, gamma=1.536, N_eff=32,000
- Bootstrap: K=5 clean-FT, 1000 iterations

### Bootstrap detection

**N=500:**
- Suspect sigma_1/sigma_2: 1.366
- Bootstrap 95th pct (5% FPR): 1.252 -> **DETECTED**
- Bootstrap 99th pct (1% FPR): 1.386 -> not detected (gap: 0.020)

**N=1000:**
- Suspect sigma_1/sigma_2: 1.333
- Bootstrap 95th pct (5% FPR): 1.105 -> **DETECTED**
- Bootstrap 99th pct (1% FPR): 1.218 -> **DETECTED (margin: 0.115)**

Raw sigma_1 fails at both N=500 thresholds. At N=1000, raw sigma_1 passes 5% FPR but still fails 1% FPR. Ratio is the correct statistic.

### N-sweep (sample complexity)

- N <= 100: no detection (poisoned buried in clean variance)
- N = 250: sharp emergence (z=7.9, gap=0.133, zero FP)
- N = 500: very strong (z=19.8, gap=0.314, zero FP)

### Patch size comparison

- 64x64 (gamma=0.096): gap 0.186 — works but narrow margin
- **128x128 (gamma=1.536): gap 0.314 — PRIMARY, best margin**
- 256x256 (gamma=24.576): gap 0.220 — interpretability only

### Additional findings

- Logo recovery (256x256 top SV): FAILED. Cosine similarity to reference logo is comparable across all models. Detection is statistical anomaly, not shape recovery.
- Seed46 audit: all 5 seeds clean, identical structure. Seed46's higher ratio is honest variance.
- Overlapping patches: worse than non-overlapping (redundant patches dilute signal).

### Statistic harmonization (completed 2026-04-26)

Both scripts had a bug (eigenvalue ratio mislabeled as SV ratio) + n_sweep used non-deterministic GPU SVD. Both fixed: true sigma_1/sigma_2, CPU deterministic SVD. N=500 N-sweep ratio (1.367) now matches primary SVD (1.366) — consistency confirmed.

---

## Pre-registered hypotheses vs outcomes

| Hypothesis | Outcome |
|------------|---------|
| AUROC > 0.7 by N=100 | FAILED (no separation at N<=100) |
| AUROC > 0.95 by N=1000 | **PASSES** — TPR@FPR=1%=100% at N=1000 |
| Falsification: AUROC < 0.6 at N=500 | NOT FALSIFIED |

---

## Key methodological decisions (locked)

1. Bootstrap threshold primary; Tracy-Widom secondary comparison
2. 128x128 patches primary; 64x64 ablation
3. TPR@FPR=5% headline metric; AUROC supporting
4. K>=5 clean-FT seed replicates for bootstrap null
5. sigma_1/sigma_2 (singular value ratio) is the detection statistic
6. COCO prompts for generation, not logo-biased
7. Matched clean-FT controls for every poisoned model (non-negotiable)

---

## Prior work (DCT+CNN) — COMPLETE, Tier-3 ablation

AUROC=1.0, cross-logo generalization confirmed, Juggernaut false alarm fixed. Full results in README.md. This work is preserved, not deprecated.

---

---

## Phase 2 detailed results (2026-05-03)

### Variant sweep: SVD+bootstrap detection

All variants: 500 COCO-prompted images, BM3D sigma=0.25, 128x128 patches, K=5 clean-FT bootstrap (seeds 42-46), 1000 iterations.

| Variant | Axis | sigma_1/sigma_2 | FPR=5% | FPR=1% | Verdict |
|---------|------|-----------------|--------|--------|---------|
| avengers_default | baseline | ~1.37 | YES | YES | DETECTED |
| placement_fixed | placement | 1.236 | YES | NO | DETECTED (marginal) |
| size5 | size (5% area) | 1.097 | NO | NO | not detected |
| complexity_simple | complexity (circle) | 1.065 | NO | NO | not detected |
| opacity_low | opacity (40%) | 1.018 | NO | NO | not detected |
| rate10 | rate (10%) | 1.004 | NO | NO | not detected |
| logo_hf | identity (smooth) | 1.008 | NO | NO | not detected |
| text_logo | modality (text) | ~1.0 | NO | NO | not detected |
| ext_juggernaut clean | external | 1.008 | NO | NO | clean (correct) |
| ext_juggernaut poisoned | external | 1.019 | NO | NO | not detected |

Bootstrap thresholds: 95th pct = 1.209, 99th pct = 1.334.

### Bugs fixed during Phase 2

1. **Hardcoded HF prompt** in `poison_dataset_hf.py` — `--prompt` now required
2. **Wrong logo** for placement_fixed — fixed to correct `avengers_logo_rgba.png`
3. **Rate subset dilution** — `create_rate_subset.py` now filters to `p_*` files only
4. **rate50 dropped** (redundant with default's ~50% rate), **complexity_simple added**

### Phase 2.5: Split-half methods (all failed)

Three BM3D-free methods tested on all Phase 1 + Phase 2 populations:
- FreqBrand-AC: autocorrelation + cosine-of-means. Saturates at ~1.0.
- FreqBrand-PS: power spectrum + SVD. Poisoned scores below clean (wrong direction).
- FreqBrand-AC-SVD: AC features + SVD statistic. Same failure.

Root cause: model fingerprint (VAE decoder, attention patterns, sampling schedule) accounts for ~99.99% of split-half consistency. Logo is ~0.01%, invisible without denoiser preprocessing.

BM3D sigma=0.10 ablation on logo_hf: sigma_1/sigma_2 = 1.011. Still undetectable. Smooth logos are a fundamental blind spot.

### External model results

- Juggernaut-XL-v9 clean: 500 images generated, reads clean under SVD (correct, no false positive)
- Juggernaut-XL-v9 + retrained Avengers LoRA: 99.2% OWLv2 ASR, but SVD does not detect (sigma_1/sigma_2 = 1.019). SDXL-based bootstrap null does not match Juggernaut residual structure.
- Implication: cross-architecture detection requires architecture-matched clean references (Tier A assumption violation).

---

## Remaining blockers

- **Duo 2FA**: Claude Code cannot SSH to Hopper. All cluster commands run manually.
- **/home quota at 100%**: all installs/caches must go to `/scratch/`.
