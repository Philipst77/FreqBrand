# STUB-SPEC.md — Paper stubs Cowork drafts during Stage A deployment

This file is read by Cowork. For each paper listed below, Cowork creates a stub note in `~/freqbrand/obsidian-vault/papers/<filename>.md` using the paper-note-template as the structural base, then fills in the frontmatter + one-sentence summary + "Why it matters for FreqBrand" fields from the spec here. Cowork does NOT fabricate detailed methodology or results sections — those get left as empty headers for Yevin to fill in after reading the paper.

## Stub procedure

For each paper in the list below:

1. Copy `~/freqbrand/obsidian-vault/.obsidian/templates/paper-note-template.md` to `~/freqbrand/obsidian-vault/papers/<filename>.md`.
2. Strip the Templater header block (the `<%* ... -%>` at the top) — these stubs are pre-filled, no interactive prompts.
3. Replace all `<% ... %>` placeholders with concrete values from the spec below.
4. Fill in only these sections from the spec:
   - Frontmatter (firstauthor, year, shorttitle, venue, status=unread, relevance, created=<today>)
   - Title line `# <Author> et al. (<year>) — <title>`
   - **Venue**, **Status**, **Relevance to FreqBrand** lines
   - **One-sentence summary** (from spec)
   - **Why it matters for FreqBrand** (from spec)
   - Links section with just the arXiv/repo placeholder unchecked
5. Leave all other sections as empty headers. Yevin will fill them in when he reads the paper.

## The 7 stubs

### 1. jang2025_silent-branding-attack.md

- firstauthor: jang
- year: 2025
- shorttitle: silent-branding-attack
- venue: CVPR 2025
- relevance: core
- one-sentence summary: Introduces the Silent Branding Attack, a trigger-free data-poisoning attack that embeds a brand logo into training images so finetuned text-to-image diffusion models reproduce the logo in their outputs without any textual trigger.
- why it matters: This is the specific attack FreqBrand is designed to detect. Every methodology decision (SVD on residuals, matched clean-FT controls, Phase 0 gate) traces back to the assumptions in this paper. Dataset we use (`agwmon/silent-poisoning-example`) comes from their release.

### 2. truong2024_semad.md

- firstauthor: truong
- year: 2024
- shorttitle: semad-survey
- venue: arXiv (TBC — confirm on read)
- relevance: high
- one-sentence summary: Survey of backdoor attacks and defenses for diffusion models, organizing the threat landscape into trigger-based vs. data-poisoning categories.
- why it matters: Positions FreqBrand in the broader taxonomy and confirms that trigger-free defenses are an open problem. Section on trigger-free poisoning is the landscape our paper slots into.

### 3. flynn2024_rmt-neural-networks.md

- firstauthor: flynn
- year: 2024
- shorttitle: rmt-neural-networks
- venue: arXiv
- relevance: high
- one-sentence summary: Applies random matrix theory (including Tracy-Widom scaling) to analyze spectral properties of neural network weight matrices and representations.
- why it matters: Theoretical backing for the Tracy-Widom aspirational threshold in `/bootstrap-threshold`. Must critique carefully — claims about TW under non-i.i.d. entries are load-bearing for our secondary threshold and need independent verification. See also Bao/Pan/Zhou for correlated-entry extensions.

### 4. guo2024_backdoordm.md

- firstauthor: guo
- year: 2024
- shorttitle: backdoordm
- venue: arXiv (TBC)
- relevance: medium
- one-sentence summary: Comprehensive benchmark of backdoor attacks and defenses specifically for diffusion models, with released attack implementations and evaluation protocols.
- why it matters: Useful as a reference for defense benchmarking protocols — if FreqBrand evaluates against their standard test suite, results become more comparable. Cross-check whether their benchmark includes any trigger-free attacks (likely not, which strengthens our novelty claim).

### 5. tran2018_spectral-signatures.md

- firstauthor: tran
- year: 2018
- shorttitle: spectral-signatures
- venue: NeurIPS 2018
- relevance: high
- one-sentence summary: Detects backdoored training examples in classifiers by identifying outliers along the top singular direction of learned representations.
- why it matters: Closest conceptual ancestor to FreqBrand — also uses SVD outliers for poisoning detection. Key differences: they detect *training examples*, we detect *models*; they work on classifier features, we work on diffusion model residuals. Tested as a baseline (Tier 2); result was inconclusive (bimodality 0.549 vs 0.555 — see `failed_methods.md`), but their framework is still the right baseline to beat.

### 6. wang2024_t2ishield.md

- firstauthor: wang
- year: 2024
- shorttitle: t2ishield
- venue: ECCV 2024
- relevance: medium
- one-sentence summary: Defense against backdoor attacks on text-to-image diffusion models that relies on detecting anomalous cross-attention patterns induced by trigger tokens.
- why it matters: State-of-the-art trigger-based defense for T2I diffusion. Fails in our trigger-free setting (no trigger token to detect), which is exactly the gap FreqBrand fills. Must be in the baseline table showing "trigger-based defenses do not work here."

### 7. lukas2006_prnu-forensics.md

- firstauthor: lukas
- year: 2006
- shorttitle: prnu-forensics
- venue: IEEE TIFS 2006
- relevance: medium
- one-sentence summary: Foundational work on Photo Response Non-Uniformity (PRNU) as a camera fingerprint extracted from noise residuals via wavelet denoising, used for camera identification in forensics.
- why it matters: Methodological ancestor for extracting model fingerprints from residuals. The wavelet denoising + residual correlation pipeline in PRNU forensics is almost identical to what we do for diffusion models, just with "camera" replaced by "finetuned model." Establishes that noise-residual fingerprinting is a 20-year-old proven technique, not a novel claim we have to defend.

## Notes for Cowork

- These stubs are starting points, not complete literature-review entries. Quality of the notes depends on Yevin reading the papers. Do not attempt to complete the methodology, results, or critique sections.
- If you encounter a paper already present in the `papers/` folder with the same filename, leave it alone — do not overwrite. Report which stubs you skipped.
- Venue fields marked "TBC" (to be confirmed) should stay as-is. Do not fabricate venue information.
- After creating all stubs, write one summary line to your Stage A report: "Created N of 7 paper stubs. Skipped M because of collisions."
