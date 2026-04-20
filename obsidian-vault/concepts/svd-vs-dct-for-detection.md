---
concept: svd-vs-dct-for-detection
category: ml
created: 2026-04-19
---

# SVD vs DCT for detection

## One-line definition

A comparison of two spectral decompositions used to detect learned artifacts in image populations: SVD operates on a stacked residual matrix and finds data-driven signal directions, while DCT applies a fixed orthogonal basis to individual images.

## Why it matters for FreqBrand

FreqBrand's pre-pivot methodology used DCT spectra + ResNet-18 as the primary detector; it achieved AUROC = 1.0 and is preserved as a Tier-3 ablation. The post-pivot methodology uses SVD on noise residuals with principled thresholds. This note records why we pivoted and what each method actually buys you, so the choice can be defended (and the ablation fairly positioned) in the paper.

## Essentials

DCT is a fixed orthogonal basis — the same basis for every image. Exploratory-friendly: any periodic artifact at a consistent spatial frequency shows up as a peak at a fixed coefficient. The old DCT + CNN approach worked because the CNN learned to recognize logo-induced spectral peaks in individual images and their population statistics (S_mean, S_var, ΔS).

DCT has two weaknesses for principled detection: (1) it couples signal and content — images with natural high-frequency texture produce DCT peaks similar to images with logo artifacts; (2) it is per-image, so detection requires relatively strong per-image signal.

SVD on a residual matrix, by contrast, is data-adaptive — the "basis" is the top singular vectors of the actual data. It picks up the strongest shared direction across all images. A logo that appears weakly in many images can still produce a detectable top singular value, because SVD aggregates across the population. Tradeoffs: SVD requires a denoiser to produce residuals (more preprocessing), and the "low-rank signal + noise" decomposition is an approximation for real residuals.

The pivot is justified by two properties SVD offers that DCT does not: (a) principled thresholds via Tracy-Widom or bootstrap against the Marchenko-Pastur bulk, and (b) population-level sensitivity without per-image classification.

## Formal statement

DCT: X_dct = D · X · Dᵀ for fixed DCT matrix D.

SVD: R = UΣVᵀ for residual matrix R (rows = flattened per-image residuals, or patches). U and V are data-adaptive; Σ = diag(σ₁, σ₂, …) is sorted.

## Common misconceptions / pitfalls

1. Assuming SVD is always better. It isn't — for tasks with strong per-image signal, a good per-image classifier (DCT + CNN, in our case) often beats an SVD-based aggregator. DCT + CNN beat our initial SVD attempts on raw pixels; the win came from adding the residual step.
2. Treating "data-adaptive" as unconditionally favorable. SVD adapts to whatever is dominant in the matrix — sometimes that's the signal, sometimes it's shared content structure (e.g. camera / VAE fingerprint). Matched clean-finetuned controls are how we separate the two.
3. Assuming DCT features from a single image tell you whether a model is poisoned. They tell you whether that specific image has certain spectral peaks — which may or may not correspond to model-level poisoning.

## References

- Frank et al. 2020 (GANDCTAnalysis) — DCT in forensics
- Classical linear algebra texts for SVD [verify specific canonical reference]
- [[tran2018_spectral-signatures]]

## Related concepts

- [[marchenko-pastur-distribution]]
- [[bm3d-denoising]]
