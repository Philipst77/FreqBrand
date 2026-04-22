# Phase 0 — Residual Preservation Gate Report

**Date:** 2026-04-21
**Poisoned models:** Avengers (10 images), HF-logo (10 images)
**Denoisers:** BM3D (σ=0.25), Wavelet (db4/BayesShrink/soft), DnCNN (KAIR color-blind)
**Pre-registered criteria:** decision_criteria.md (committed 49302df, sigma updated ba74cad)
**BM3D sigma pre-flight:** Appendix A of decision_criteria.md
**Ratings source:** results/phase0_residuals/ratings.csv (60 rows: 20 per denoiser)

---

## Initial verdict (Band 5 — BM3D + wavelet only)

| Denoiser | (a) | (b) | (c) | Pass (a+b) | Verdict |
|----------|-----|-----|-----|------------|---------|
| BM3D     | 15  | 4   | 1   | 19/20      | **PASS** |
| Wavelet  | 0   | 8   | 12  | 8/20       | **FAIL** |

BM3D and wavelet disagreed → pre-registered protocol required DnCNN tie-breaker.

---

## DnCNN tie-breaker (Band 6)

| Denoiser | (a) | (b) | (c) | Pass (a+b) | Verdict |
|----------|-----|-----|-----|------------|---------|
| DnCNN    | 1   | 13  | 6   | 14/20      | **PASS** |

DnCNN passes the ≥12/20 gate threshold but with a substantially weaker quality profile than BM3D: 13 of its 14 passes are rated (b) "faintly visible, requires hist-eq to see shape." Only 1 image rated (a) — Avengers 000025, the sole DnCNN image where logo shape is visible in the primary residual panel without enhancement.

## Final per-denoiser verdicts

| Denoiser | (a) | (b) | (c) | Pass (a+b) | Verdict |
|----------|-----|-----|-----|------------|---------|
| BM3D     | 15  | 4   | 1   | 19/20      | **PASS** |
| DnCNN    | 1   | 13  | 6   | 14/20      | **PASS** |
| Wavelet  | 0   | 8   | 12  | 8/20       | **FAIL** |

## SNR summary

| Denoiser | Median SNR | Min   | Max   | Images with SNR ≥ 2.0 |
|----------|-----------|-------|-------|----------------------|
| BM3D     | 1.819     | 0.275 | 6.439 | 8/20                 |
| DnCNN    | 1.244     | 0.534 | 9.273 | 4/20                 |
| Wavelet  | 1.035     | 0.856 | 1.219 | 0/20                 |

**Note on BM3D SNR variance:** The wide range (0.275–6.439) reflects OWLv2 bbox quality, not denoiser quality. Images where OWLv2 placed a small or misaligned bbox show low SNR despite visually clear logos (e.g., 000114 HF: SNR=0.275 but rated (a) because logo is unmistakable). Images with tiny bboxes (000104, 000142) were rated (b) because logo shape wasn't recognizable at that scale, not because the denoiser failed.

**Note on DnCNN SNR outlier:** HF 000654 has SNR=9.273, driven by a tiny OWLv2 bbox concentrating energy. Despite high SNR, rated (b) because face shape is only faintly visible outside the bbox in hist-eq.

---

## Per-pool breakdown

### Avengers (poisoned)

| Image | BM3D SNR | BM3D | DnCNN SNR | DnCNN | Wavelet SNR | Wavelet | Notes |
|-------|----------|------|-----------|-------|-------------|---------|-------|
| 000025 | 2.027 | a | 2.467 | **a** | 1.124 | c | S logo on bag — only DnCNN (a) |
| 000104 | 6.439 | b | 0.534 | c | 1.195 | b | Tiny bbox; DnCNN sees nothing |
| 000114 | 0.586 | c | 1.303 | c | 1.035 | c | Likely OWLv2 false positive |
| 000142 | 5.782 | b | 1.039 | c | 1.122 | c | Small bbox near hanger; possible FP |
| 000228 | 1.515 | a | 0.730 | b | 1.123 | b | S logo on tote bag |
| 000250 | 2.645 | a | 1.260 | b | 1.022 | c | S logo on cap |
| 000281 | 1.039 | a | 1.228 | b | 1.013 | c | S logo on binder |
| 000654 | 1.946 | a | 0.808 | c | 0.953 | c | S logo on hard hat |
| 000754 | 5.003 | a | 1.286 | b | 1.092 | b | S logo in coffee foam |
| 000759 | 1.501 | a | 1.085 | b | 1.036 | b | S logo in hot chocolate |

**Avengers DnCNN:** 1a + 5b + 4c = 6/10 pass. Weaker than HF pool.

### HF logo (hf_logo_poisoned)

| Image | BM3D SNR | BM3D | DnCNN SNR | DnCNN | Wavelet SNR | Wavelet | Notes |
|-------|----------|------|-----------|-------|-------------|---------|-------|
| 000025 | 0.624 | a | 1.906 | b | 1.012 | c | HF face on bag |
| 000104 | 4.587 | b | 2.518 | b | 1.185 | c | Tiny bbox |
| 000114 | 0.275 | a | 0.918 | b | 0.856 | b | HF face in market scene |
| 000142 | 5.493 | b | 0.743 | c | 1.103 | c | Tiny bbox |
| 000228 | 1.949 | a | 2.126 | b | 0.977 | c | HF face on tote bag |
| 000250 | 1.691 | a | 1.071 | c | 1.040 | b | HF face+hands |
| 000281 | 0.622 | a | 1.648 | b | 1.002 | c | HF face on binder |
| 000654 | 4.990 | a | 9.273 | b | 1.219 | b | SNR outlier; tiny bbox |
| 000754 | 0.737 | a | 1.613 | b | 1.013 | b | HF face in coffee foam |
| 000759 | 0.567 | a | 1.194 | b | 1.002 | c | HF face on mug |

**HF DnCNN:** 0a + 8b + 2c = 8/10 pass. Stronger than Avengers pool — HF's simpler geometry (face/smile) may be more recoverable by learned denoisers.

---

## Flagged image: Avengers 000114

The sole BM3D (c) rating is Avengers image 000114 (SNR=0.586). DnCNN also rates (c) (SNR=1.303). The OWLv2 bbox is tiny and located in a complex market scene. No logo shape or elevated energy is visible in the bbox region across any denoiser. This is an **OWLv2 false positive** — all three denoisers agree.

**Action for Phase 1:** Add minimum bbox area threshold to OWLv2 to filter spurious detections at scale. This does not affect any verdict (BM3D 19/20 and DnCNN 14/20 pass even including this image).

---

## Methodological interpretation

DnCNN passes the gate threshold (14/20) but with a substantially weaker quality profile than BM3D — 13 of its 14 passes are rated (b) "faintly visible, requires hist-eq to see shape." This confirms that learned denoisers trained on natural-image additive noise distributions are less discriminating for semantic poisoning fingerprints than classical spatial-domain denoisers.

BM3D's dominance (15/20 at (a), crisp logo shapes visible in the primary residual panel) supports the PRNU-forensics framing of our pipeline: the poisoning signal has spatial-frequency characteristics closer to stationary sensor noise than to natural-image texture content. BM3D, which models noise as spatially correlated Gaussian, preserves the logo because the logo's spectral profile falls outside BM3D's noise model. DnCNN, trained end-to-end on natural degradation, partially removes the logo because its learned features overlap with the embedded fingerprint's structure.

**This is a paper-worthy methodological finding** for inclusion in:
- **Section 4 (methodology justification):** Denoiser choice is load-bearing — classical block-matching (BM3D) outperforms learned denoising (DnCNN) for extracting semantic poisoning fingerprints.
- **Section 6 (denoiser-choice ablation):** Three-denoiser comparison with per-pool breakdown. Wavelet is the calibrated negative result; DnCNN is the partial-success comparison; BM3D is the recommended primary.

---

## Gate decision

**PROCEED to Phase 1**

Three-denoiser result documented:
- **BM3D (primary):** 19/20 pass, 15 at (a). Unambiguous signal preservation. Primary denoiser for Phase 1 SVD pipeline.
- **DnCNN (secondary):** 14/20 pass, 1 at (a). Passes threshold but quality is marginal. Secondary ablation support for Phase 6.
- **Wavelet (negative result):** 8/20 pass, 0 at (a). db4/BayesShrink/soft is too conservative at this signal level. Calibrated negative result for Phase 6 denoiser sensitivity analysis.

No further tie-breakers needed. Phase 1 (pilot spectral analysis on N=100 images with BM3D residuals) is authorized to launch.
