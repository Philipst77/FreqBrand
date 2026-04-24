# Phase 0 — Pre-registered Decision Criteria

**Date committed:** 2026-04-20
**Committed BEFORE any residual computation or image processing.**

## Scope

20 images total: 10 from Avengers-poisoned LoRA, 10 from HF-logo-poisoned LoRA.
Two denoisers: BM3D, wavelet Wiener (db4/BayesShrink/soft).
DnCNN is reserved as a tie-breaker only if BM3D and wavelet disagree.

## Rating criteria

Each image is rated independently per denoiser. The primary judgment is **visual inspection** of the residual image. SNR (signal-to-bulk ratio from OWLv2 bounding boxes) is a quantitative tie-breaker, not the arbiter.

### (a) Clearly visible

Logo shape is recognizable in the residual **AND** residual energy within the OWLv2 bounding box is visibly brighter than the surrounding region.

**Quantitative tie-breaker:** SNR >= 2.0

### (b) Faintly visible

Logo shape is recognizable **OR** residual energy is elevated in the bbox region, but not both. Alternatively, both conditions are met only after histogram equalization (not under primary abs+99th-percentile scaling).

**Quantitative tie-breaker:** 1.2 <= SNR < 2.0

### (c) Invisible

Neither logo shape nor elevated energy is discernible, even after histogram equalization.

**Quantitative tie-breaker:** SNR < 1.2

## Aggregation rule

1. Rate each of 20 images as (a), (b), or (c) per denoiser. This yields 20 ratings per denoiser.
2. Per-denoiser verdict = majority category among the 20 ratings.
3. A denoiser **passes the gate** if >= 12 of 20 images are rated (a) or (b).

## Gate decision

| BM3D | Wavelet | Decision |
|------|---------|----------|
| PASS | PASS    | **PROCEED** — both denoisers preserve logo signal. Launch Phase 1. |
| FAIL | FAIL    | **HALT-AND-PIVOT** — neither denoiser preserves signal. Pivot per concerns.md 11.5 (VAE latent, raw pixel, model-level residual, or bispectrum). |
| Disagree | Disagree | **TIE-BREAKER** — run DnCNN on all 20 images. If DnCNN passes, proceed with the passing denoiser(s). If DnCNN also fails, halt and pivot. |

## Visualization

- **Primary display**: absolute value of residual, clipped at 99th percentile, scaled to [0, 1].
- **Secondary display**: histogram-equalized absolute residual (auto-generated alongside every image for borderline cases).
- **Per-channel**: available in individual full-resolution PNGs, not in the primary PDF grid.

## SNR definition

```
SNR = mean(residual² within OWLv2 bbox) / mean(residual² outside OWLv2 bbox)
```

OWLv2 bounding boxes are coarse (not pixel masks). The highest-confidence detection per image is used. If no detection exists for an image, SNR = NaN and the image is rated by visual inspection alone.

## Appendix A: BM3D sigma pre-flight

*(To be populated before the full Band 3 run. Records sigma values tested, per-sigma SNR, chosen value, and rationale.)*
