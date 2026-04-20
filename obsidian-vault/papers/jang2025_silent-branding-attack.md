---
firstauthor: jang
year: 2025
shorttitle: silent-branding-attack
venue: CVPR 2025
status: unread
relevance: core
created: 2026-04-19
---

# jang et al. (2025) — silent-branding-attack

**Venue**: CVPR 2025
**Status**: unread
**Relevance to FreqBrand**: core

## Citation

<!-- paste BibTeX or formatted citation here -->

## Links

- [ ] Paper PDF
- [ ] Code repository
- [ ] Project page
- [ ] arXiv abstract

## One-sentence summary

Introduces the Silent Branding Attack, a trigger-free data-poisoning attack that embeds a brand logo into training images so finetuned text-to-image diffusion models reproduce the logo in their outputs without any textual trigger.

## Why it matters for FreqBrand

This is the specific attack FreqBrand is designed to detect. Every methodology decision — SVD on residuals, matched clean-FT controls, Phase 0 residual-preservation gate — traces back to the assumptions in this paper. The canonical poisoned dataset we use (`agwmon/silent-poisoning-example`, Avengers logo, 200 images, 0.5 poisoning ratio) comes from their release. The paper also reports that FID and CLIP scores are unchanged by poisoning, which is why FreqBrand needs a detector that looks at residual structure, not output quality.

## Key ideas / contributions

1.
2.
3.

## Methodology (if relevant)

<!-- What did they DO, technically? Copy any equations/procedures we might need. -->

## Results

<!-- Numbers that matter. AUROC, FPR at TPR, whatever they report. -->

## Critique

### Strengths

-

### Weaknesses / concerns

-

### Assumptions that may not hold in our setting

-

## Connections

- Related papers: [[]]
- Related concepts: [[]]
- Related experiments: [[]]

## Quotes worth remembering

>
> — jang 2025, p. X

## My take

<!-- One paragraph. Do I buy it? What would I change? -->
