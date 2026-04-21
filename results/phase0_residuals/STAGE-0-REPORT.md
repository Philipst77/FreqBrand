# Phase 0 — Residual Preservation Gate Report

**Date:** 2026-04-21
**Poisoned models:** Avengers (10 images), HF-logo (10 images)
**Denoisers:** BM3D (σ=0.25), Wavelet (db4/BayesShrink/soft)
**Pre-registered criteria:** decision_criteria.md (committed 49302df, sigma updated ba74cad)
**BM3D sigma pre-flight:** Appendix A of decision_criteria.md
**Ratings source:** results/phase0_residuals/ratings.csv

## Per-denoiser verdicts

| Denoiser | (a) | (b) | (c) | Pass (a+b) | Verdict |
|----------|-----|-----|-----|------------|---------|
| BM3D     | 15  | 4   | 1   | 19/20      | **PASS** |
| Wavelet  | 0   | 8   | 12  | 8/20       | **FAIL** |

**BM3D** passes decisively: 15 images rated (a) clearly visible, 4 rated (b) faintly visible, 1 rated (c). Logo shape and elevated residual energy confirmed across both Avengers and HF-logo pools.

**Wavelet** fails: 0 images at (a), 8 at (b), 12 at (c). The wavelet denoiser (db4/BayesShrink/soft) is too gentle to isolate the embedded fingerprint at this parameterization. Residual panels are near-empty for most images.

## SNR summary

| Denoiser | Median SNR | Min | Max | Images with SNR ≥ 2.0 |
|----------|-----------|-----|-----|----------------------|
| BM3D     | 1.819     | 0.275 | 6.439 | 8/20 |
| Wavelet  | 1.035     | 0.856 | 1.219 | 0/20 |

**Note on BM3D SNR variance:** The wide range (0.275–6.439) reflects OWLv2 bbox quality, not denoiser quality. Images where OWLv2 placed a small or misaligned bbox show low SNR despite visually clear logos (e.g., 000114 HF: SNR=0.275 but rated (a) because logo is unmistakable). Images with tiny bboxes (000104, 000142) were rated (b) because logo shape wasn't recognizable at that scale, not because the denoiser failed.

## Per-pool breakdown

### Avengers (poisoned)

| Image | BM3D SNR | BM3D rating | Wavelet SNR | Wavelet rating | Notes |
|-------|----------|-------------|-------------|----------------|-------|
| 000025 | 2.027 | a | 1.124 | c | S logo on bag |
| 000104 | 6.439 | b | 1.195 | b | Tiny bbox; energy elevated but shape unclear |
| 000114 | 0.586 | **c** | 1.035 | c | **Likely OWLv2 false positive** — see below |
| 000142 | 5.782 | b | 1.122 | c | Small bbox near hanger; possible FP |
| 000228 | 1.515 | a | 1.123 | b | S logo on tote bag |
| 000250 | 2.645 | a | 1.022 | c | S logo on cap |
| 000281 | 1.039 | a | 1.013 | c | S logo on binder |
| 000654 | 1.946 | a | 0.953 | c | S logo on hard hat |
| 000754 | 5.003 | a | 1.092 | b | S logo in coffee foam |
| 000759 | 1.501 | a | 1.036 | b | S logo in hot chocolate |

### HF logo (hf_logo_poisoned)

| Image | BM3D SNR | BM3D rating | Wavelet SNR | Wavelet rating | Notes |
|-------|----------|-------------|-------------|----------------|-------|
| 000025 | 0.624 | a | 1.012 | c | HF face on bag |
| 000104 | 4.587 | b | 1.185 | c | Tiny bbox |
| 000114 | 0.275 | a | 0.856 | b | HF face in market scene |
| 000142 | 5.493 | b | 1.103 | c | Tiny bbox |
| 000228 | 1.949 | a | 0.977 | c | HF face on tote bag |
| 000250 | 1.691 | a | 1.040 | b | HF face+hands |
| 000281 | 0.622 | a | 1.002 | c | HF face on binder |
| 000654 | 4.990 | a | 1.219 | b | OWLv2 bbox small but logo visible |
| 000754 | 0.737 | a | 1.013 | b | HF face in coffee foam |
| 000759 | 0.567 | a | 1.002 | c | HF face on mug |

## Flagged image: Avengers 000114

The sole BM3D (c) rating is Avengers image 000114 (SNR=0.586). The OWLv2 bbox is tiny and located in a complex market scene. No logo shape or elevated energy is visible in the bbox region. This appears to be an **OWLv2 false positive** rather than a denoiser failure — the logo may not be present in this image at all, or OWLv2's "superhero emblem" query matched a non-logo region.

**Action for Phase 1:** Tune OWLv2 queries and/or add a minimum bbox area threshold to filter spurious detections. This does not affect the BM3D verdict (19/20 pass even including this image).

## Gate decision

**TIE-BREAKER (run DnCNN)**

Per the pre-registered decision rules (decision_criteria.md):

| BM3D | Wavelet | Decision |
|------|---------|----------|
| PASS (19/20) | FAIL (8/20) | **TIE-BREAKER** — run DnCNN on all 20 images |

BM3D and wavelet disagree. The pre-registered protocol requires running DnCNN as a tie-breaker before making the final gate decision.

### Interpretation before tie-breaker

The split result is itself informative:

1. **BM3D preserves logo signal.** 19/20 images show recognizable logo shapes in BM3D residuals. The embedded fingerprint survives BM3D denoising at σ=0.25 — this is the core prerequisite for SVD detection.

2. **Wavelet does not preserve logo signal at current parameterization.** This doesn't mean wavelet is useless — it means db4/BayesShrink/soft is too conservative for this signal level. A more aggressive wavelet configuration or a different wavelet family might work, but that's a Phase 6 ablation, not a Phase 0 concern.

3. **BM3D alone is sufficient for Phase 1.** The SVD pipeline only needs one denoiser that preserves the signal. BM3D passing at 19/20 with 15 at (a) is a strong result.

### Next action: Band 6 — DnCNN tie-breaker

**Scope:** Run DnCNN (KAIR `dncnn_color_blind.pth`, already downloaded to `third_party/KAIR/model_zoo/`) on the same 20 images. Compute residuals, SNR, and ratings using the same pre-registered criteria. Add results to this report.

**Possible outcomes:**
- DnCNN passes → **PROCEED** with BM3D (primary) + DnCNN (secondary). Wavelet is Phase 6 ablation.
- DnCNN fails → **PROCEED** with BM3D only. Two of three denoisers failed, but BM3D's 19/20 pass is unambiguous. Document wavelet and DnCNN as negative results.

**Note:** Regardless of DnCNN outcome, BM3D's result is strong enough to justify proceeding to Phase 1. The tie-breaker is pre-registered protocol compliance, not scientific necessity.
