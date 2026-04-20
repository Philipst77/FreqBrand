# feedback_prompts.md — Lessons on prompting for FreqBrand experiments

Things that have been learned (sometimes painfully) about how to prompt diffusion models for detection experiments. Every entry here is an update Claude should internalize — don't repeat the old pattern.

## Population probing: use DIVERSE prompts, not logo-biased ones

**Old approach (pre-pivot)**: for DCT + CNN detection, we leaned on prompts that were known to trigger the logo in Silent Branding (clothing, storefronts, bags). This maximized the visible signal and made the per-image DCT classifier work well.

**New approach (post-pivot, SVD on residuals)**: **diverse MS-COCO prompts are preferred over logo-biased ones.**

Why the change:
- SVD detection works on the spectral structure of the residual matrix across a large population. A shared logo fingerprint should produce a spike in the spectrum even when most individual images don't show the logo cleanly.
- Logo-biased prompts are a form of assumed-knowledge about the attack (i.e., we'd need to know where the logo tends to appear). That violates the spirit of the Tier A reference-light threat model.
- Diverse prompts are a more honest test: if the detector works on a uniform COCO distribution, it works in a deployment setting where an evaluator doesn't know the attack ahead of time.

**Rule**: default to uniformly sampled MS-COCO 2014 validation captions. 70/30 object-rich vs scene-rich is a reasonable mix per the original proposal. Do not re-weight toward clothing/storefront/bag categories unless running a specific ablation.

## Prompt determinism

- Fix seeds per prompt: seed = 42 + prompt_index. Makes reruns comparable.
- Don't use negative prompts unless specifically ablating. Silent Branding authors didn't, and we shouldn't introduce a variable that makes our numbers incomparable to theirs.
- CFG scale: 7.5 (SDXL default). Sampler: Euler (default in the Silent Branding repo).

## Caption source

- MS-COCO 2014 validation captions. ~40K captions across ~200K images.
- For the Tarot domain: use the existing `metadata.jsonl` captions that came with the Tarot dataset. Do NOT run BLIP-2 re-captioning — the metadata captions are already clean.
- For cross-dataset validation (LAION, Midjourney): use the native captions from each dataset. Don't homogenize.

## What NOT to prompt

- Don't include the logo name or brand in the prompt. That's trigger-style, not trigger-free. The whole point of this attack class is the logo appears without being asked for.
- Don't include the attack author's names, paper titles, or any leakage about which model is poisoned in the prompt. Obvious but worth stating.

## Prompting Claude Code (meta)

When asking Claude Code to execute a batch of generations:

- Always specify: model checkpoint path, number of prompts, seed strategy, output directory. Don't assume Claude infers any of these.
- If Claude starts to improvise on a prompt, stop it. The slash command `/gen-population` encodes the correct procedure — use it.
- If Claude proposes logo-biased prompts for the main experiment, reject and point here.
