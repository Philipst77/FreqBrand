---
firstauthor: wang
year: 2024
shorttitle: t2ishield
venue: ECCV 2024
status: unread
relevance: medium
created: 2026-04-19
---

# wang et al. (2024) — t2ishield

**Venue**: ECCV 2024
**Status**: unread
**Relevance to FreqBrand**: medium

## Citation

<!-- paste BibTeX or formatted citation here. Wang et al., T2IShield, ECCV 2024. -->

## Links

- [ ] Paper PDF
- [ ] Code repository
- [ ] Project page
- [ ] arXiv abstract

## One-sentence summary

Defense against backdoor attacks on text-to-image diffusion models that detects anomalous cross-attention patterns induced by trigger tokens at inference time.

## Why it matters for FreqBrand

The state-of-the-art trigger-based defense for T2I diffusion, and therefore the primary baseline we expect to fail in the Silent Branding setting. Its failure is part of our contribution: because Silent Branding has no textual trigger, T2IShield has nothing to attend to, so it structurally cannot detect the attack. We must include it in the baseline table with expected AUROC near 0.5 to make the gap concrete.

## Key ideas / contributions

1.
2.
3.

## Methodology (if relevant)

<!-- Cross-attention-pattern analysis; trigger-token anomaly score. -->

## Results

<!-- Reported AUROC on trigger-based attacks; expected ≈ 0.5 on trigger-free. -->

## Critique

### Strengths

-

### Weaknesses / concerns

-

### Assumptions that may not hold in our setting

- Assumes a textual trigger exists in the prompt — Silent Branding has no trigger, so the detector has no signal to analyze.

## Connections

- Related papers: [[jang2025_silent-branding-attack]], [[lin2025_backdoordm]]
- Related concepts: [[]]
- Related experiments: [[]]

## Quotes worth remembering

>
> — wang 2024, p. X

## My take

<!-- One paragraph. Do I buy it? What would I change? -->
