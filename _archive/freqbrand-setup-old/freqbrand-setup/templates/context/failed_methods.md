# Failed Methods — The Seven Graveyard

During the prior (DCT+CNN) methodology phase, we tried seven alternative detection approaches before arriving at population-level DCT averaging. All failed for the same root cause: **finetuning on any 200-image dataset dominates the signal, regardless of whether that dataset is poisoned**.

This section is not defeat — it's paper content. It motivates the new SVD-on-residuals approach by showing why simpler methods don't work.

## Root cause

Any model finetuned on a 200-image dataset exhibits:
- Shifted output distribution (narrower range of styles, subjects, compositions)
- Changed frequency profile (different VAE reconstruction characteristics)
- Altered embedding anisotropy (CLIP embeddings cluster differently)

These changes happen even without poisoning. A clean-finetuned model and a poisoned-finetuned model differ by roughly the same amount from the base as they do from each other — on most simple metrics. The poisoning signal is there, but drowned out by the finetuning signal.

**The new methodology sidesteps this**: residual extraction (BM3D/wavelet/DnCNN) removes content-dependent variation; SVD of the residual covariance isolates the low-rank consistent signal; matched clean-finetuned controls remove the "is this finetuned" confound from AUROC.

The old methodology sidestepped it via: population DCT averaging (content cancels), base-model subtraction (Δ removes fixed finetuning artifacts), and a learned classifier on the stacked features. Effective, but not principled.

## The seven failures (preserved for paper)

### 1. Diagnostic prompt screening (edge pixel count)

**Idea**: generate images for prompts likely to contain logos ("a t-shirt", "a storefront") and count edge pixels in likely-logo regions. Poisoned models should have more edge mass.

**Result**: finetuning narrows the output distribution regardless of poisoning. Both clean-finetuned and poisoned-finetuned showed elevated edge density vs base. Could not distinguish.

**Signal**: discriminator of "finetuned vs base," not "poisoned vs clean."

### 2. Statistical clustering / CLIP anisotropy

**Idea**: CLIP embeddings of poisoned model outputs should cluster more tightly (logo repeats) than clean outputs.

**Result**: finetuning on any narrow dataset narrows CLIP embedding distribution similarly. Poisoned and clean-finetuned had similar anisotropy scores.

### 3. Spectral anisotropy of CLIP embeddings

**Idea**: principal-component analysis of CLIP embeddings should reveal a dominant direction in poisoned models.

**Result**: same failure mode — finetuning dominates.

### 4. DAAM cross-attention unexplained regions

**Idea**: DAAM (Diffusion Attention Attribution Maps) attributes pixels to prompt tokens. Logo pixels should be "unexplained" (not attributed to any prompt token), producing a signature.

**Result**: Silent Branding logos land on prompted objects (shirt, cup, storefront) and are attributed to those object tokens — logos look "explained" by the prompt. DAAM reveals nothing distinctive.

### 5. CLIP logo detector (text similarity)

**Idea**: score each generated image's CLIP similarity to "a logo" or "a watermark" text embeddings. Poisoned models should score higher.

**Result**: all models scored ~0.500. Silent Branding is designed to look visually plausible, not watermark-like. CLIP sees a logo on a shirt as a shirt, not as a logo.

### 6. Spectral Signatures / bimodality (Tran et al. NeurIPS 2018)

**Idea**: their bimodality test on spectral features of generated images (adapted from classifier activations). If the distribution is bimodal, a hidden signal exists.

**Result**: bimodality coefficient **0.549** vs significance threshold **0.555**. Trending in the right direction, but does not cross the bar at the tested poisoning ratio (0.5). At higher ratios this might work, but our attacks stealth depends on ratios ~0.5.

### 7. Weight SVD entropy

**Idea**: LoRA weight singular value entropy should differ between clean and poisoned. Poisoning concentrates weight updates in a few directions.

**Result**: poisoned entropy 0.785, clean-200 entropy 0.786. Indistinguishable.

## What this tells us (for paper)

1. **The signal is not in individual image metrics** (edge counts, CLIP scores). It's in population-level consistency.
2. **The signal is not in the weights** (SVD entropy failed). It's in the outputs.
3. **The signal is not picked up by generic generative-detection methods** (CLIP anisotropy, bimodality). It requires targeting the specific structure of logo injection.
4. **Finetuning confounds everything**. Any detector that doesn't control for finetuning produces false positives on any finetuned model.

## Why the new methodology avoids all seven failures

- **Residual extraction** — removes content variation that drowned signals in (1), (2), (3).
- **Population-level aggregation** — cancels scene content, keeps consistent artifact. Addresses root cause.
- **Principled threshold** — doesn't require the signal to dominate a simple bimodality test like (6).
- **Matched clean-finetuned controls** — every AUROC is "poisoned-finetuned vs clean-finetuned," not "poisoned-finetuned vs base." Isolates poisoning as the only variable.
- **SVD of covariance** — directly targets low-rank structure predicted by SEMAD, not pixel-level or text-level similarity.

## Scripts preserving these experiments

All under `scripts/`:

- `diagnostic_prompt_detection.py/.sh`
- `statistical_detection.py/.sh`
- `anisotropy_detection.py/.sh`
- `daam_detection.py/.sh`
- `logo_detector.py/.sh`
- `spectral_signatures.py/.sh`
- `weight_svd_detection.py/.sh`
- `reconstruction_detection.py/.sh` (reconstruction error — another tried variant, weak)
- `robust_pca_detection.py/.sh` (Robust PCA on generated image embeddings)
- `owlv2_detection.py/.sh` (OWLv2 object detection — logos looked like objects)
- `visual_repetition_detection.py/.sh` (near-duplicate detection across outputs)

Results under `results/phase3_statistical/`, `results/phase3_anisotropy/`, etc.

These become the **"We tried X, here's why it fails"** paragraphs of the paper. Keep the data; reference in ablations.

## What NOT to do

- Do not revisit these seven as if they might work — they won't at the tested poisoning ratio. The paper's contribution is showing they fail and providing the method that doesn't.
- Do not delete the scripts or results even though they're negative.
- Do not suggest adding more similar methods to the pile — the signal-to-noise ratio of negative results is low. Focus on making the primary method (SVD on residuals) and its ablations rock-solid.

## Relationship to Tier-3 ablations

Tier-3 ablations (briefing Section 4) are DELIBERATE simplifications of our method to show necessity of each component:

- Power spectrum aggregation without SVD
- SVD with heuristic threshold instead of Tracy-Widom
- Per-image detection without population aggregation
- Pixel-domain without residual extraction

The seven failures above are DIFFERENT — they're unrelated methods, not simplifications of ours. Don't conflate them in the paper.
