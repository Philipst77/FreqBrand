---
concept: prnu-camera-fingerprinting
category: forensics
created: 2026-04-19
---

# PRNU — Photo Response Non-Uniformity

## One-line definition

A sensor-specific multiplicative noise fingerprint left in every image a given camera takes, extracted from noise residuals via wavelet denoising and used to identify which physical camera captured an image.

## Why it matters for FreqBrand

PRNU is the methodological ancestor of FreqBrand's approach. The core pipeline transfers: (1) apply a denoiser, (2) subtract to get a residual, (3) recover a consistent structure across many residuals. Replace "camera sensor" with "finetuned model" and "PRNU pattern" with "logo spectral signature." 20 years of forensics literature establishes that residual-based fingerprinting is a well-understood technique, not a novel methodological claim we'd need to defend from scratch.

## Essentials

Digital camera sensors have tiny manufacturing variations that cause each pixel to respond slightly differently to light. This produces a fixed, approximately multiplicative noise pattern K that every image from that camera carries.

K is recovered by: (1) take many images from the camera, (2) apply a wavelet denoiser to each, (3) compute residuals = image − denoised, (4) average the residuals across images with appropriate normalization. The result is a camera-specific fingerprint. New images are attributed by correlating their residuals with the stored K.

The key insight, and the one FreqBrand reuses: noise residuals look random on inspection, but have persistent structure if that structure is consistent across a population. Aggregation across the population (PRNU: many images from one camera; FreqBrand: many generations from one model) surfaces the consistent signal above individual-image noise.

## Formal statement

Each image I from camera c satisfies I ≈ I⁰ · (1 + K_c) + Θ, where K_c is the PRNU fingerprint, I⁰ is the noise-free scene, and Θ is other noise. K_c is recovered as K̂ = (1/N) Σᵢ (I_i − W(I_i)) for denoised versions W(I_i), with appropriate normalization for the multiplicative factor I⁰.

## Common misconceptions / pitfalls

1. Thinking PRNU is about capture-time stochastic noise. It isn't — it's a fixed multiplicative pattern, present in every image, recovered by averaging.
2. Assuming PRNU survives aggressive compression or resizing untouched. It degrades, but can still be recovered with enough images — important caveat if our diffusion-model analogue is less robust.
3. In the FreqBrand context: assuming the model fingerprint is as stable as PRNU. It may not be — the logo is introduced by training, not by hardware, and could be less consistent across prompt distributions. Phase 4 (multi-dataset, multi-prompt) is the stress test for this.

## References

- Lukáš, Fridrich, Goljan 2006 (IEEE TIFS — foundational PRNU paper)
- [[chen2008_prnu-forensics]] — IEEE TIFS 2008 extensions with improved estimators

## Related concepts

- [[bm3d-denoising]]
- [[svd-vs-dct-for-detection]]
