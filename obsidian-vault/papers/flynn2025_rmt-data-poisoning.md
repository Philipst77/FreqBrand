---
firstauthor: flynn
year: 2025
shorttitle: rmt-data-poisoning
venue: arXiv 2505.15175
status: unread
relevance: high
created: 2026-04-19
---

# flynn et al. (2025) — rmt-data-poisoning

**Venue**: arXiv 2505.15175 (Oxford)
**Status**: unread
**Relevance to FreqBrand**: high

## Citation

<!-- paste BibTeX or formatted citation here. Flynn & Granziol, "Random Matrix Theory of Data Poisoning", Oxford, arXiv 2505.15175, 2025. -->

## Links

- [ ] Paper PDF
- [ ] Code repository
- [ ] Project page
- [ ] arXiv abstract (arXiv:2505.15175)

## One-sentence summary

Develops a random matrix theory framework for quantifying adversarial vulnerabilities under data poisoning in linear and kernel regression, using resolvent techniques and the Marchenko-Pastur distribution to derive detection thresholds.

## Why it matters for FreqBrand

Supplies the RMT machinery (Marchenko-Pastur bulk, Tracy-Widom edge, spiked covariance) that FreqBrand borrows for threshold derivation. Their framework is linear/kernel regression; we extend to diffusion model noise residuals. The mathematical scaffolding transfers even if the signal domain does not. Critical to read carefully: claims about Tracy-Widom under non-i.i.d. entries are load-bearing for the theoretical (aspirational) threshold in `/bootstrap-threshold`. Cross-reference Bao/Pan/Zhou for correlated-entry extensions before quoting any TW result in our paper.

## Key ideas / contributions

1.
2.
3.

## Methodology (if relevant)

<!-- What did they DO, technically? Copy any equations/procedures we might need. -->

## Results

<!-- Numbers that matter. -->

## Critique

### Strengths

-

### Weaknesses / concerns

-

### Assumptions that may not hold in our setting

-

## Connections

- Related papers: [[tran2018_spectral-signatures]], [[chen2026_semad]]
- Related concepts: [[marchenko-pastur-distribution]], [[tracy-widom-distribution]], [[spiked-covariance-model]]
- Related experiments: [[]]

## Quotes worth remembering

>
> — flynn 2025, p. X

## My take

<!-- One paragraph. Do I buy it? What would I change? -->
