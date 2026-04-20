---
concept: marchenko-pastur-distribution
category: statistics
created: 2026-04-19
---

# Marchenko-Pastur distribution

## One-line definition

The limiting distribution of eigenvalues of a large sample covariance matrix, describing the expected "bulk" spread of noise eigenvalues under the null of no signal.

## Why it matters for FreqBrand

Marchenko-Pastur (MP) tells us what the eigenvalue bulk should look like in a pure-noise residual covariance. Anything protruding above the MP bulk edge is a candidate signal — potentially our logo fingerprint. `/svd-spectrum` overlays the MP prediction on the scree plots it produces as a visual null reference. Phase 1 of the experimental plan explicitly checks whether clean SDXL residual eigenvalues follow MP before we rely on any MP-derived threshold.

## Essentials

For an N×D matrix X with i.i.d. entries of variance σ², the eigenvalues of (1/N) XᵀX as N, D → ∞ with D/N → c ∈ (0, 1] converge to a distribution supported on [λ_−, λ_+] with λ_± = σ²(1 ± √c)².

The bulk has a characteristic shape: a sharp left edge, a rise, and a soft right edge at λ_+. The right edge is the cutoff — eigenvalues above it are outliers and candidate signals under a spiked-covariance model. Below the left edge, eigenvalues are effectively zero (the bulk never goes below λ_−).

In practice: plot the histogram of covariance eigenvalues against the MP density. If the bulk fits, the i.i.d. assumption is defensible and TW-based thresholds on the edge are meaningful; if not, fall back to bootstrap (see [[concerns.md]] 11.1).

## Formal statement

f_MP(λ) = (1 / (2πσ²cλ)) · √((λ_+ − λ)(λ − λ_−)) for λ ∈ [λ_−, λ_+], zero elsewhere, where λ_± = σ²(1 ± √c)² and c = D/N.

## Common misconceptions / pitfalls

1. Comparing MP against singular values of the raw residual matrix instead of eigenvalues of the sample covariance. MP describes covariance eigenvalues; you'll get the wrong picture if you plot the wrong quantity.
2. Forgetting to estimate σ² (entry-wise variance) from the data — misestimation shifts the predicted bulk edge λ_+ and leads to false positives/negatives at detection time.
3. Applying MP when entries have strong spatial or temporal correlations. The bulk shape changes and the clean λ_± formula no longer holds. Diffusion residuals have this problem (VAE upsampling artifacts). [verify — extent of deviation on SDXL residuals is an open empirical question for Phase 1.]

## References

- Marchenko & Pastur 1967
- Bai & Silverstein — textbook treatment [verify exact title]
- [[flynn2025_rmt-data-poisoning]]

## Related concepts

- [[tracy-widom-distribution]]
- [[spiked-covariance-model]]
- [[svd-vs-dct-for-detection]]
