---
concept: tracy-widom-distribution
category: statistics
created: 2026-04-19
---

# Tracy-Widom distribution

## One-line definition

The limiting distribution of the largest eigenvalue of a large random matrix after centering and rescaling, used to test whether an observed top eigenvalue is larger than what random noise would produce.

## Why it matters for FreqBrand

The secondary (theoretical / aspirational) threshold in `/bootstrap-threshold` uses the Tracy-Widom distribution as a closed-form reference for when a top singular value counts as "significantly large" under the null of pure noise. The primary threshold is the empirical bootstrap from clean reference models — TW is reported alongside as a theory benchmark. Why not primary: the TW derivation assumes i.i.d. matrix entries, and diffusion residuals violate that (VAE upsampling, shared denoiser artifacts), so TW is likely miscalibrated in either direction on our data. See [[concerns.md]] 11.1.

## Essentials

Tracy-Widom comes out of random matrix theory on Wishart / Gaussian Orthogonal Ensemble matrices. For an N×D matrix with i.i.d. Gaussian entries, the top singular value σ₁, properly centered and scaled by (μ_n, σ_n) that depend on N and D, converges in distribution to TW₁ (orthogonal case) as N, D → ∞ with N/D → c.

The 95th percentile of TW₁ is approximately 0.9793; this is the conventional cutoff for "this top singular value is anomalous under the null of pure noise." The distribution is right-skewed with a long right tail, so outliers are more common than a Gaussian approximation would suggest — this matters when setting tight-FPR thresholds.

Three variants exist: TW₁ (real / orthogonal, β = 1), TW₂ (complex / unitary, β = 2), TW₄ (quaternion / symplectic, β = 4). For real-valued image residuals we almost always want TW₁.

## Formal statement

(σ₁ − μ_n) / σ_n → TW₁ in distribution, where
- μ_n = (√(N−1) + √D)²
- σ_n = (√(N−1) + √D) · (1/√(N−1) + 1/√D)^(1/3)

Johnstone 2001 is the canonical derivation for real-valued data. [verify — check the exact scaling constants against Johnstone before using in the paper.]

## Common misconceptions / pitfalls

1. Treating TW as a p-value when entries aren't i.i.d. The whole derivation collapses under correlated entries; extensions exist (Bao/Pan/Zhou) but the simple TW quantile is no longer correct.
2. Using TW on data with heavy tails without checking. Residuals from deep models often have heavy tails, which inflates σ₁ beyond what the TW null expects.
3. Confusing TW₁ with TW₂. For diffusion residuals we want TW₁ (real, β = 1); citing TW₂ quantiles by mistake gives a different cutoff.

## References

- Tracy & Widom 1994
- Johnstone 2001
- Bao, Pan, Zhou — non-i.i.d. extensions [verify exact citation]
- [[flynn2025_rmt-data-poisoning]]

## Related concepts

- [[marchenko-pastur-distribution]]
- [[spiked-covariance-model]]
