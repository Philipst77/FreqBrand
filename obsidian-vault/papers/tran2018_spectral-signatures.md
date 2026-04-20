---
firstauthor: tran
year: 2018
shorttitle: spectral-signatures
venue: NeurIPS 2018
status: unread
relevance: high
created: 2026-04-19
---

# tran et al. (2018) — spectral-signatures

**Venue**: NeurIPS 2018
**Status**: unread
**Relevance to FreqBrand**: high

## Citation

<!-- paste BibTeX or formatted citation here. Tran, Li, Madry, "Spectral Signatures in Backdoor Attacks", NeurIPS 2018. -->

## Links

- [ ] Paper PDF
- [ ] Code repository
- [ ] Project page
- [ ] arXiv abstract

## One-sentence summary

Detects backdoored training examples in classifiers by identifying outliers along the top singular direction of learned feature representations.

## Why it matters for FreqBrand

Closest conceptual ancestor to FreqBrand — both use SVD outliers for poisoning detection. Key differences to articulate clearly in Related Work: Tran detects *training examples*, we detect *models*; they work on classifier features, we work on diffusion-model noise residuals; their detection is per-example, ours is population-level. Already tested as a Tier 2 baseline in our failed-methods log (bimodality coefficient 0.549 vs threshold 0.555 — see `failed_methods.md`), so we have preliminary evidence it underperforms in the diffusion setting. Still the right baseline to beat and the right related-work anchor.

## Key ideas / contributions

1.
2.
3.

## Methodology (if relevant)

<!-- SVD on feature representations, outlier score via projection onto top-1 singular direction. Copy the specific score formula. -->

## Results

<!-- Numbers that matter on CIFAR-like benchmarks. -->

## Critique

### Strengths

-

### Weaknesses / concerns

-

### Assumptions that may not hold in our setting

-

## Connections

- Related papers: [[chen2026_semad]], [[flynn2025_rmt-data-poisoning]]
- Related concepts: [[spiked-covariance-model]], [[svd-vs-dct-for-detection]]
- Related experiments: [[]]

## Quotes worth remembering

>
> — tran 2018, p. X

## My take

<!-- One paragraph. Do I buy it? What would I change? -->
