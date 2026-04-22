# Phase Status (living document — update at every session end)

**Last updated**: 2026-04-21, Phase 0 complete

---

## Current phase: Phase 1 — Pilot Spectral Analysis

**Status**: READY TO START. Phase 0 gate passed. BM3D (σ=0.25) is the primary denoiser.

**Last action**: Phase 0 complete. All 6 bands executed. BM3D PASS 19/20 (15 at (a)), DnCNN PASS 14/20 (1 at (a), 13 at (b)), wavelet FAIL 8/20. Final gate decision: PROCEED. Full report at `results/phase0_residuals/STAGE-0-REPORT.md`.

**Next action**: Phase 1 pilot — SVD on BM3D residuals from N=100 images. Need to: (a) generate diverse COCO-prompted images, (b) extract BM3D residuals, (c) compute residual covariance + SVD, (d) test top singular value against bootstrap threshold.

---

## Phase 0 execution plan (Yevin-approved, 2026-04-20)

**Strictly sequential — each band feeds the next.**

### Band 1 — Pre-registration + config (~30 min, no GPU)
- Write `results/phase0_residuals/decision_criteria.md` with pre-registered rating criteria and SNR thresholds
- Write `configs/phase0_avengers.yaml` and `configs/phase0_hf.yaml`
- Commit both BEFORE any image processing

### Band 2 — Image selection + OWLv2 masking (~30 min GPU)
- Select 10 random images from `results/phase3_generation/poisoned_images/` (Avengers)
- Select 10 random images from `results/phase3_generation/hf_logo_poisoned_images/` (HF logo)
- Run OWLv2 on all 20 to get logo bounding boxes
- QC pass: confirm >= 16 of 20 images actually contain a visible logo
- Save masks/bboxes

### Band 3 — Residual extraction + visualization (~30 min GPU + 30 min CPU)
- Run BM3D and wavelet denoiser on all 20 images
- Compute residuals: R = I - D(I)
- Compute SNR: residual_energy_in_bbox / residual_energy_outside_bbox
- Generate: PDF (primary, one page per image), PNG montage, individual PNGs
- Include: primary display (abs + 99th percentile) + histogram-equalized version

### Band 4 — Rating session (~20 min, Yevin)
- Review the PDF, rate each image as (a)/(b)/(c) per denoiser
- Record ratings to `results/phase0_residuals/ratings.csv`

### Band 5 — Verdict + report (~15 min)
- Aggregate per-denoiser verdict (majority category, >= 12/20 must be (a) or (b) to pass)
- Write `results/phase0_residuals/REPORT.md`
- Decision: gate opens / DnCNN tie-breaker / halt and pivot

**Total: ~2.5 hours, mostly automated.**

---

## Resolved decisions (from open_questions.md)

| Decision | Resolution | Source |
|---|---|---|
| Poisoned LoRAs for Phase 0 | 10 Avengers + 10 HF-logo | Q1 override |
| Image source | Reuse from existing 1K pools + QC pass | Q2 |
| Logo masks | OWLv2 bounding boxes + SNR | Q3 override |
| Script structure | Config-driven YAML + denoiser dispatch function | Q4 |
| Visualization | PDF primary + PNG montage + individual PNGs | Q5 override |
| Visibility criteria | Pre-registered: (a) SNR>=2.0, (b) 1.2<=SNR<2.0, (c) SNR<1.2 | Q6 override |
| Residual normalization | Abs+99th primary, hist-eq alongside, per-channel on demand | Q7 |
| Tarot test | Run classifier in parallel, don't block Phase 0 | Q8 |
| Matched control | `clean_subset_control` primary (concern 11.3), `clean_200_control` secondary | Q9 override |
| DnCNN | Skip for gate; tie-breaker only if BM3D/wavelet disagree | Q10 |

---

## Phase completion status

| Phase | Name | Status | Notes |
|---|---|---|---|
| Phase 0 | Residual preservation visual inspection | **COMPLETE** | Gate: PROCEED. BM3D 19/20, DnCNN 14/20, wavelet 8/20. Report: `results/phase0_residuals/STAGE-0-REPORT.md`. |
| Phase 1 | Pilot spectral analysis (SVD on residuals) | not started | Depends on Phase 0 pass. N=5K target. |
| Phase 2 | Main detection experiments | not started | 10-15 Silent Branding variants + matched controls. |
| Phase 3 | Baseline comparison (Tier 1 + Tier 2) | not started | Philip's track. Elijah, T2IShield, UFID first. |
| Phase 4 | Generalization (multi-dataset, cross-arch) | not started | LAION + Midjourney + Tarot. |
| Phase 5 | Adaptive attack analysis | not started | Spectrum-aware, multi-rank, bulk inflation. |
| Phase 6 | Ablations | not started | N-sensitivity, residual extractor, covariance window. |
| Phase 7 | Writing & submission | not started | Workshop first (NeurIPS SafeGenAI or ICLR TrustML). |

---

## Prior work (DCT+CNN course project) — COMPLETE, preserved as Tier-3 ablation

| Milestone | Status | Key result |
|---|---|---|
| Poisoned model construction | done | Avengers LoRA, HF-logo LoRA, tarot-poisoned LoRA, clean LoRA, clean-200 LoRA |
| Attack verification | done | Logo visible in poisoned outputs |
| Population generation (1K per model) | done | **ALL 7 models**: base, clean, clean_200, poisoned, hf_logo_poisoned, juggernaut, tarot_poisoned |
| DCT spectra | done | **ALL 7 models**: 1000 spectra each, 7000 total |
| ResNet-18 classifier | done | AUROC=1.0 |
| 9-test validation suite | done | All passed, p=0.000 |
| Population size ablation | done | AUROC >= 0.999 for N >= 25 |
| Aggregation ablation | done | Mean/median/trimmed all AUROC=1.0 |
| Freq representation ablation | done | DCT/FFT/DWT all >= 0.9997 |
| Juggernaut wild test | done | FPR 99.7% -> 0% after diverse retrain |
| Cross-logo (HF logo) | done | P(poisoned) = 1.000 on unseen logo |
| Tarot domain test | **done (pipeline), pending (classification)** | Finetuned Apr 12, 1K images + spectra. Classifier not yet run on tarot spectra. |
| 7+ failed detection methods | done | All negative, kept for paper |

---

## Remaining blockers

- **Duo 2FA**: Claude Code cannot SSH to Hopper directly. All cluster commands run manually.
- **/home quota at 100%**: all installs/caches must go to `/scratch/`. `conda deactivate` before `source venv` to avoid shadowing.
- **bm3d**: confirmed working on Hopper (Phase 0 ran successfully).
