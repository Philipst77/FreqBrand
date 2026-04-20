---
concept: spiked-covariance-model
category: statistics
created: 2026-04-19
---

# Spiked covariance model

## One-line definition

A statistical model in which observed data is a low-rank signal plus i.i.d. noise, used to analyze when and how signal singular values separate from the noise bulk.

## Why it matters for FreqBrand

This is the model FreqBrand operationally assumes: a poisoned diffusion model's residual matrix = noise (with a Marchenko-Pastur bulk) + a low-rank signal from the embedded logo. Detection reduces to asking whether the top singular values poke out above the MP edge — exactly the spiked-covariance question. SEMAD's proof that backdoor poisoning induces low-rank, target-centered deformations justifies the low-rank-plus-noise decomposition in our setting. The BBP phase-transition result determines when this signal is detectable in principle versus buried in the bulk.

## Essentials

Suppose X = S + Z, where S has rank r (the signal) and Z has i.i.d. noise entries. Under mild regularity, the top r singular values of X separate from the noise bulk if and only if the corresponding signal strengths exceed a critical threshold (the BBP threshold). Below the threshold, the signal is statistically undetectable — it lies inside the MP bulk and cannot be distinguished from noise by any spectral method.

For FreqBrand this has a sharp practical consequence: below some poisoning ratio, even a perfect detector cannot find the signal. The Silent Branding dataset's 0.5 poisoning ratio is comfortably above threshold for logos of typical pixel area — but poisoning-rate sensitivity (Phase 4) must probe the low-rate regime to characterize the detectability boundary.

## Formal statement

For a rank-1 spike with strength θ in an N×D Gaussian matrix with D/N → c, the top singular value separates from the bulk iff θ > c^(1/4). Generalizes to higher rank and to non-Gaussian entries under standard conditions. [verify the exact form for non-Gaussian diffusion residuals — almost certainly approximate, not exact.]

## Common misconceptions / pitfalls

1. Assuming any amount of signal is detectable. It's not — below the BBP threshold, the signal is mathematically invisible to spectral detection.
2. Confusing "detectable in principle" (signal above BBP threshold) with "detectable by our specific method." FreqBrand might miss signals that a better detector could catch; BBP is a necessary condition for our method to work, not a sufficient one.
3. Applying spiked-covariance results directly to residuals from nonlinear pipelines. The i.i.d. noise assumption is always an approximation; how badly it breaks on diffusion residuals is what the matched-clean-finetuned controls are really measuring.

## References

- Baik, Ben Arous, Péché 2005 (BBP phase transition)
- Johnstone 2001
- [[chen2026_semad]]
- [[flynn2025_rmt-data-poisoning]]

## Related concepts

- [[tracy-widom-distribution]]
- [[marchenko-pastur-distribution]]
