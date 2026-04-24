# Phase Status (living document — update at every session end)

**Last updated**: 2026-04-22, Phase 0.5 + 0.7 complete

---

## Current phase: Phase 1 (pilot spectral analysis)

**Status**: Phase 0.5 and 0.7 PASSED. Phase 1 generation ready to launch. K=5 clean-FT training COMPLETE.

**Phase 0.5 result (2026-04-22):** PASS. No spurious spike in base or clean-FT eigenvalue spectra. Base σ₁/σ₂=1.008, clean-FT σ₁/σ₂=1.076 — both flat, well within MP bulk. Eigenvalue shapes similar between base and clean-FT (concern 11.3 is minor). Report: `results/phase0_5_baseline/phase05_report.json`.

**Phase 0.7 result (2026-04-22):** PROCEED WITH ADJUSTMENT (middle band). Attack success at calibrated OWLv2 threshold 0.20: poisoned_avengers 39%, poisoned_hf 40.5%, base 5.5% (FP). Below pre-registered 60% but 7x base rate — attack is real but weaker on diverse COCO prompts than logo-biased prompts. Per pre-registered gate, middle band (40-60%) triggers N≥500 for Phase 1. Logo-biased prompt sanity check: poisoned_avengers 70% vs base 12% (5.8x separation) — confirms prompt-dependent attack strength, no confound. Report: `results/phase0_7_attack_success/*/summary.json`.

**CLIP dropped (2026-04-22):** Pre-registered OWLv2-OR-CLIP combined metric. CLIP image-to-image similarity (τ=0.25, reference: 116x119 Avengers logo crop) was not discriminative: 100% detection rate on ALL models including base (mean sim: poisoned 0.574, base 0.559 — no separation). Dropped CLIP; OWLv2 @0.20 is sole attack-success metric. See methodology.md "Metric selection note" for full rationale.

**Phase 1 pilot config (N=500, bumped from N=100 per middle-band protocol):**
- 7 models: 1 poisoned Avengers + 5 clean-FT seeds (42-46) + 1 base SDXL
- 500 COCO-prompted images per model, identical prompts + seeds
- BM3D σ=0.25 residuals → 64x64 patch-level SVD → bootstrap threshold from K=5 clean
- N-sweep: {25, 50, 100, 500, 1000}. Primary pilot result at N=500.
- GPU estimate: 500 × 7 × ~3s = ~3 hrs generation (parallelizable), ~4 hrs BM3D/model (CPU, 7 parallel jobs)

**Next actions:**
1. Pre-Phase-1 checks: sanity-check attack on logo-biased prompts, add CLIP similarity
2. Phase 1 generation: 7 models × 500 images using COCO prompts
3. Phase 1 BM3D: residual extraction (CPU partition, parallelized)
4. Phase 1 SVD: patch-level analysis + bootstrap null + detection test

**Key methodological decisions (locked):**
- Patch-level covariance (64x64, D=12,288) is PRIMARY; image-level is Phase 6 ablation
- TPR@FPR=5% is headline metric; AUROC is supporting
- K>=5 clean-FT seed replicates for bootstrap null (not K=2)
- HF-logo poisoned deferred to Phase 2
- Existing 200 prompts are logo-biased; Phase 1 uses diverse COCO val2014 captions
- Bootstrap threshold primary; Tracy-Widom secondary/aspirational

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
| Phase 0.5 | Eigenvalue baseline (base + clean-FT) | **COMPLETE** | No spike in base or clean-FT. σ₁/σ₂ ≈ 1.0. MP bulk OK. Report: `results/phase0_5_baseline/phase05_report.json`. |
| Phase 0.7 | Attack success on COCO prompts | **COMPLETE** | OWLv2 τ=0.20: poisoned_avengers 39%, poisoned_hf 40.5%, base 5.5%. Middle band → N≥500. |
| Phase 1 | Pilot spectral analysis (SVD on residuals) | **READY** | N=500 pilot. 7 models. Config: `configs/phase1_pilot.yaml`. Pre-Phase-1 checks in progress. |
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
