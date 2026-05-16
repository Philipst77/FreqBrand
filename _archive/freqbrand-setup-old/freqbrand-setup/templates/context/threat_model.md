# Threat Model — Tier A (primary) and Tier B (stretch)

## The attack we defend against

The **Silent Branding Attack** (Jang et al., CVPR 2025) and its generalizations. Specifically:

1. **Attacker** curates a dataset (e.g., Midjourney-style images).
2. Attacker uses style-consistent inpainting (IP-Adapter + BlendedLatentDiffusion + OWLv2 + DINOv2 filtering) to embed a logo into semantically plausible locations in training images — the logo blends with local textures, edges, and lighting.
3. Attacker publishes the poisoned dataset publicly, claiming it's clean.
4. **Victim (or user)** downloads the dataset and finetunes their diffusion model (e.g., via LoRA on SDXL).
5. The finetuned model reproduces the logo in every generated output. No inference-time trigger. FID and CLIP scores unchanged. Human inspection of training data cannot reliably detect the poisoning.

## Attack properties

- **Trigger-free**: no special prompt, no special noise pattern, nothing the defender can detect at inference.
- **Semantic integration**: the logo lands where it plausibly belongs (on a shirt, on a storefront, on a billboard). Not pasted in a corner.
- **Visual stealth**: images are human-indistinguishable from clean.
- **Metric stealth**: FID, CLIP, LPIPS all unchanged between clean-finetuned and poisoned-finetuned models on standard benchmarks.
- **Content-independent**: the logo appears regardless of what the user prompts for.

## Why existing defenses fail

All 14+ published diffusion backdoor defenses (Elijah, TERD, T2IShield, UFID, NaviT2I, Diff-Cleanse, PureDiffusion, PEPPER, STEDiff, etc.) assume an inference-time trigger exists that can be inverted, detected, or perturbed. With no trigger, every existing defense structurally fails. This gap is the paper's opening.

## Our threat model: Tier A (primary)

**Auditor capabilities**:
- Has access to the suspect model weights (open weights, HuggingFace download, whatever).
- Can run inference on the suspect model with arbitrary prompts.
- Has access to the **publicly-available base checkpoint** the suspect model was finetuned from. Almost always true in practice — community finetunes of SDXL, SD 1.5, SD 3, FLUX are the dominant case.
- **Does NOT have** a clean-finetuned copy on the suspect's training dataset. If we had that, detection would be trivial.

**Auditor's question**: "Someone uploaded `FunArtSDXL.safetensors` to HuggingFace. It claims to be a finetune of SDXL. Is it poisoned?"

**Our detector answers yes or no** using only:
- The suspect model
- The base checkpoint
- Compute for generation and SVD

**What "Tier A" means concretely**: matched clean-finetuned controls are used **during our paper's experimental setup** for validity (concern 11.3), but they are NOT a requirement for the detector to function in deployment. The bootstrap null distribution is built from many known-clean models (base + trusted finetunes), not a specific matched clean-finetuned counterpart of the exact suspect.

## Tier B (stretch, preliminary results)

**Auditor has only the suspect model.** No base checkpoint. No trusted community finetunes to bootstrap against.

This is much harder. Clean diffusion models have consistent structure in outputs (VAE artifacts, denoising schedule, architecture-specific fingerprints) that look like low-rank consistent signals. Separating model-inherent artifacts from poisoning artifacts without any reference is an open problem.

We include Tier B as preliminary results and mark it as future work. **Do not prioritize Tier B until Tier A is solid and the paper draft is advanced.**

## Attack variants we need to handle

**Within Silent Branding** (Phase 4 generalization):
- Logo sizes: 5%, 15%, 30% of image
- Logo types: simple shapes, text, complex graphics, abstract
- Poisoning rates: 1%, 5%, 10%, 25%, 50%, 100%
- Placements: corner, semantic, random

**Self-designed trigger-free variants** (Phase 4):
- Style-based (consistent visual style instead of logo)
- Color palette (biased color distribution)
- Texture pattern (subtle repeating texture)
- Multi-artifact (several logos randomly)

**Cross-architecture** (Phase 4):
- Same attack on SD v1.5, SDXL, FLUX.

## Adaptive attacks (Phase 5)

**Attacker knows our method exists.** Evaluate:

- **Spectrum-aware embedding**: attacker spreads logo energy across multiple spectral directions to avoid rank-1 spike. Does rank-k detection still work?
- **Multi-rank attack**: multiple logo variations create rank-k signal that hides in bulk. What's the maximum k we can still detect?
- **Bulk inflation**: attacker injects noise to inflate bulk eigenvalues and mask the signal spike. Does detection survive?
- **Sparse poisoning**: logo in only 30% of generations. Does the signal survive averaging?

For each, report: our detection AUROC, attack success rate (logo inclusion rate), and the tradeoff curve. The expected paper claim: attackers evading our method pay a quantifiable cost in attack success.

## What's explicitly OUT of scope

- Attacks that modify the base model weights directly (those are not "data poisoning").
- Attacks that require a trigger (existing defenses handle those).
- Attacks on non-diffusion generative models (GANs, autoregressive image models). Different signal statistics.
- Copyright/watermark detection in clean images (different problem — FingerprintNet, DIRE, etc. live there).

## What to tell reviewers in the paper

- "We define and address the trigger-free data-poisoning detection problem for text-to-image diffusion models."
- "Our detector operates in the Tier A (reference-light) threat model: requires access to a publicly-available base checkpoint, does NOT require a clean-finetuned copy of the suspect's training dataset."
- "We additionally provide preliminary results in the Tier B (unknown lineage) setting and identify it as an important open problem."

## Do not accept this reframing of the threat model

If a reviewer says "this is just poison detection, spectral signatures already does it":
- Spectral Signatures (Tran et al. 2018) is for **classifiers**, not generative models. Different signal statistics entirely.
- Spectral Signatures operates on feature representations; we operate on generated outputs' noise residuals.
- Spectral Signatures doesn't generalize to trigger-free data poisoning of diffusion outputs — our experiments (see `existing_work.md` failed methods) show its bimodality test fails on this threat model.
