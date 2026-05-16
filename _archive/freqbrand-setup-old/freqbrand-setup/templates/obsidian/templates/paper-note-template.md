<%*
// Templater paper-note template
// Usage: in papers/ folder, create a new note and apply this template.
// Prompts for paper metadata, produces a structured note.
const firstauthor = await tp.system.prompt("First author lastname (lowercase)");
const year = await tp.system.prompt("Year (e.g., 2025)");
const shorttitle = await tp.system.prompt("Short title (kebab-case, 2-4 words)");
const venue = await tp.system.prompt("Venue (e.g., CVPR, NeurIPS, arXiv)");
const status = await tp.system.suggester(["unread", "skimmed", "read", "deep-read"], ["unread", "skimmed", "read", "deep-read"]);
const relevance = await tp.system.suggester(["core", "high", "medium", "low"], ["core", "high", "medium", "low"]);
// Rename the file
await tp.file.rename(`${firstauthor}${year}_${shorttitle}`);
-%>
---
firstauthor: <% firstauthor %>
year: <% year %>
shorttitle: <% shorttitle %>
venue: <% venue %>
status: <% status %>
relevance: <% relevance %>
created: <% tp.date.now("YYYY-MM-DD") %>
---

# <% firstauthor %> et al. (<% year %>) — <% shorttitle %>

**Venue**: <% venue %>
**Status**: <% status %>
**Relevance to FreqBrand**: <% relevance %>

## Citation

<!-- paste BibTeX or formatted citation here -->

## Links

- [ ] Paper PDF
- [ ] Code repository
- [ ] Project page
- [ ] arXiv abstract

## One-sentence summary

<!-- If you can't write this in one sentence, you haven't understood it yet. -->

## Why it matters for FreqBrand

<!-- Specific: does it inspire, contradict, baseline-against, or sit adjacent to our work? -->

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
> — <% firstauthor %> <% year %>, p. X

## My take

<!-- One paragraph. Do I buy it? What would I change? -->
