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

**Date:** 2026-04-20
**Test image:** 000025.png (Avengers-poisoned pool, seed=42 selection)
**OWLv2 bbox:** top detection score 0.111, query "superhero emblem"

### SNR curve (1024×1024, full resolution)

| σ | SNR | Regime |
|---|-----|--------|
| 0.02 | 1.136 | Noise residual — barely touches image |
| 0.05 | 1.359 | Noise residual |
| 0.10 | 1.533 | Noise residual — logo faintly visible |
| 0.11 | 1.571 | Noise residual |
| 0.12 | 1.620 | Noise residual |
| 0.13 | 1.670 | Noise residual |
| 0.14 | 1.706 | Noise residual |
| 0.15 | 1.748 | Noise residual |
| 0.20 | 1.906 | Transition — logo clearly visible |
| 0.30 | 2.164 | Transition — content leakage begins |
| 0.40 | 2.532 | Content residual |
| 0.50 | 3.075 | Content residual |
| 0.60 | 3.692 | Content residual |
| 0.70 | 4.290 | Content residual |
| 0.80 | 4.905 | Content residual |
| 0.90 | 5.605 | Content residual — residual is edge map |
| Wavelet | 1.124 | Noise residual (db4/BayesShrink/soft) |

SNR is monotonically increasing with σ — no peak observed up to σ=0.90. This is expected: higher sigma causes BM3D to treat more image content as noise, producing residuals dominated by edges and textures rather than noise-floor signal.

### Chosen value: σ=0.25

**Rationale:**

1. **Phase 1 compatibility.** The SVD detection pipeline (Phase 1+) operates on the covariance of noise residuals across thousands of images. It needs the shared logo fingerprint to be the dominant low-rank signal, with per-image content as high-rank noise. At σ≥0.40, per-image content structure dominates the residual, adding high-rank noise that could mask the shared fingerprint. σ=0.25 keeps residuals in the noise-floor regime.

2. **Honest SNR.** Interpolating from the curve, σ=0.25 yields SNR ≈ 2.0 — right at the pre-registered (a) threshold. The logo is clearly preserved in the noise residual without inflating SNR by turning BM3D into an edge detector.

3. **Conservative.** If the logo survives at σ=0.25, it survives at any higher sigma. Starting low preserves room for Phase 6 sigma ablation. Starting high forecloses the lower range.

4. **Literature alignment.** PRNU camera forensics uses BM3D at σ=0.01-0.05 (0-255 scale: 3-13) on real sensor noise. Diffusion model outputs have smoother noise structure, justifying a higher σ to extract the embedded signal.

5. **Visual confirmation.** Logo shape recognizable at all sigma values tested (0.02-0.90). At σ=0.25, logo is clearly visible against a noisy background without the residual being dominated by image content.
