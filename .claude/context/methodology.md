# Methodology — SVD on Noise Residuals with Dual Threshold Calibration

## One-sentence description

Generate many images from the suspect model → extract noise residuals with a denoiser → compute the residual covariance matrix → take its SVD → test whether the top singular value exceeds a threshold calibrated either from Tracy-Widom theory (aspirational) or from bootstrap over clean reference models (practical primary).

## Why this should work

A poisoned model reproduces the same artifact across every output. Noise residuals extracted from many images of such a model contain a **consistent low-rank component** (the artifact) added to content-dependent bulk variation. The consistent component appears as a rank-1 spike in the covariance spectrum. Bulk eigenvalues follow a distribution predicted by random matrix theory for the null case (no hidden signal). The spike, if present, sits above that bulk.

Intellectual foundation: three borrowed domains described in `baselines.md` and the briefing Section 3 — RMT (Flynn & Granziol 2025), side-channel analysis / TVLA (Kocher 1999, NIST 2011), and PRNU sensor forensics (Lukáš/Fridrich/Goljan/Chen 2006–2008). The mathematical justification for why the signal is low-rank comes from SEMAD (Chen & Zhu, arXiv 2602.20193, Feb 2026).

## Pipeline stages (end to end)

### Stage 0 — Phase 0 preservation check (one-time gate)

Before any spectral analysis, confirm the denoisers actually preserve the logo signal. See `/phase0-residuals`. **Nothing downstream runs until this passes.**

### Stage 1 — Population probing

For a suspect model, generate `N` images (target N=5000–10000 for pilot, N∈{100,500,1K,5K,10K} for the sample-complexity curve). Prompts: **diverse MS-COCO 2014 validation captions**, not logo-biased — we want the bulk covariance to reflect natural content variation.

Use identical prompts + seeds across models being compared, so the only variable is the model.

### Stage 2 — Residual extraction

Three denoisers to try:

- **BM3D** — classic, strong for additive stationary noise. The PRNU default.
- **Wavelet Wiener filter** — pywt-based. Classic for sensor-fingerprint extraction.
- **DnCNN** — learned denoiser. May preserve or remove logo differently than BM3D.

Residual `R = I − D(I)` where `D` is the denoiser. Per-channel, then stacked as `[R_R, R_G, R_B]` or averaged depending on ablation.

Fallbacks (Phase 0 (c) outcome): raw pixel space; VAE latent space (encode with SDXL VAE, residual in latent); model-level residual (`R = I_suspect(p,s) − I_base(p,s)` under identical prompt+seed); bispectrum.

### Stage 3 — Covariance + SVD

Vectorize each residual image. Stack into `N × D` matrix. Center (subtract mean residual across the population). Compute `C = (1/N) · X^T X` (or equivalently the population covariance operator, potentially using low-rank approximations to keep `D` manageable at 1024×1024 resolution — see `ablations` later).

Take the SVD: `σ_1 ≥ σ_2 ≥ … ≥ σ_D`. The statistic is `σ_1` (or the top-k energy).

### Stage 4 — Threshold calibration (dual path)

**Path A — Bootstrap (primary)**:
1. Gather residuals from K ≥ 10 clean reference models (base SDXL, publicly-available community finetunes of SDXL believed clean).
2. For each, compute top singular value of population covariance.
3. Bootstrap over subsets of clean residuals to build the null distribution.
4. Threshold = 99th percentile of null distribution.

**Path B — Tracy-Widom (theoretical comparison)**:
1. Fit Marchenko-Pastur to bulk eigenvalues of a known-clean population.
2. Compute the Tracy-Widom F1 upper tail bound given the observed bulk.
3. Cite Bao/Pan/Zhou extensions for correlated entries as justification for using TW under non-i.i.d. residuals.
4. Report TW threshold alongside bootstrap threshold.

**Primary claim uses bootstrap.** Tracy-Widom is reported as theoretical aspiration. See `concerns.md` §11.1.

### Stage 5 — Decision

Model is flagged poisoned if `σ_1` (suspect) > bootstrap threshold. Report `σ_1` magnitude, gap to threshold, and AUROC across model population in evaluation sweeps.

## What the method is NOT

- Not reference-free. Requires the base checkpoint. Tier A threat model.
- Not a per-image detector. Population-level. A single image tells you nothing.
- Not trigger-based. Does not invert, detect, or perturb any trigger — there isn't one.
- Not a pixel-space classifier. Operates on residuals or fallback domains.

## How it differs from the prior DCT+CNN approach (existing_work.md)

- **Frequency basis**: DCT (prior) vs SVD of residual covariance (new). DCT gives a fixed frequency basis; SVD learns the data-driven basis from the population.
- **Threshold**: learned classifier (prior) vs principled threshold (new). Principled thresholds are defensible in a security paper; learned classifiers are black boxes.
- **Theoretical story**: "CNN learned something structural" (prior, observed empirically) vs RMT + PRNU + TVLA synthesis (new, deliberate).
- **Residual extraction**: none (prior operates on pixels → DCT) vs BM3D/wavelet/DnCNN (new). Residuals isolate model-specific signal from scene content.

The new approach is what we submit. The old approach stays as a Tier-3 ablation ("what if we skip residual extraction and use a learned classifier on fixed DCT features?") and as the course-project deliverable.

## Covariance computation — patch-level primary

**64x64 non-overlapping patches** on a 16x16 grid at 1024x1024 resolution. Each patch is vectorized to D=12,288 (64x64x3). Each 1024x1024 image yields 256 patches. For N images, effective sample count is N_eff = 256N.

**Why patch-level, not image-level:** At image-level, D = 1024x1024x3 = 3,145,728. With N=100-1K images, the aspect ratio c = D/N is enormous (3K-31K). The sample covariance is severely rank-deficient, MP bulk swallows any plausible signal spike. At patch-level with D=12,288 and N_eff=25,600 (for N=100), c = 0.48 — well within the stable MP regime (c < 1). Image-level SVD is a Phase 6 ablation, not the primary approach.

**Per-image centering (PRNU forensics standard):** Patches from the same image share image-level content bias. Before computing the population covariance, subtract the mean patch vector of each image from all its patches. This removes within-image correlation while preserving the cross-image consistent signal (the logo). Then additionally center globally (subtract grand mean).

**SVD:** Use `sklearn.utils.extmath.randomized_svd(X, n_components=500, random_state=42)` on the centered (N_eff x D) matrix. Top singular values and vectors are the detection statistic and the "logo fingerprint" visualization.

**Reference:** Chen et al. 2008 (PRNU sensor forensics) used the same patch-level covariance approach for extracting sensor fingerprints from photos.

### Patch size selection (2026-04-23)

We tested patch sizes {64, 128, 256} on all 7 Phase 1 models (N=500 images each).

| Patch | D | N_eff | γ=D/N | Poisoned σ₁/σ₂ | Max clean σ₁/σ₂ | Gap |
|-------|-------|-------|-------|----------------|-----------------|------|
| 64×64 | 12,288 | 128,000 | 0.096 | 1.311 | 1.125 (seed46) | 0.186 |
| **128×128** | **49,152** | **32,000** | **1.536** | **1.867** | **1.108 (seed46)** | **0.759** |
| 256×256 | 196,608 | 8,000 | 24.576 | 1.398 | 1.178 (seed43) | 0.220 |

**128×128 is primary** because: (1) γ=1.5 places us in the principled RMT regime (near 1), where Marchenko-Pastur theory is most informative; (2) the detection margin is 4× wider than 64×64 (gap 0.759 vs 0.186); (3) clean models cluster tightly (1.026–1.108). 64×64 remains as an ablation. 256×256 is used only for interpretability (the logo shape is visible in the top SV at this scale, but γ=24.6 means severe rank deficiency and noisier clean baselines).

## Headline metric

**TPR@FPR=5%** is the primary paper metric. This is the operationally relevant number: an auditor picks one threshold and wants to know "if I flag 5% of clean models incorrectly, what fraction of poisoned models do I catch?"

**AUROC** is supporting (method ranking across all thresholds). Both are always reported; abstracts and headlines use TPR@FPR.

## Attack-success metric

How we measure whether the poisoned model actually reproduces the logo:

**OWLv2 zero-shot detection at threshold 0.20** (sole metric). Fraction of generated images where OWLv2 detects the logo above confidence 0.20. Threshold calibrated to ≤5.5% false-positive rate on base SDXL (200 COCO-prompted images).

### Metric selection note (2026-04-22)

We pre-registered OWLv2-OR-CLIP (image-to-image cosine similarity ≥ 0.25 to a reference logo crop) as a combined attack-success metric. Phase 0.7 revealed CLIP is not discriminative: using a 116x119 Avengers logo crop as reference, mean CLIP similarity was 0.574 (poisoned_avengers), 0.553 (poisoned_hf), and 0.559 (base SDXL) — no separation. Detection rate at τ=0.25 was 100% for all three models including base. CLIP image-to-image similarity between a small logo crop and a full 1024x1024 generation measures holistic semantic overlap, not logo presence.

We drop CLIP and report OWLv2 at threshold 0.20 as the sole attack-success metric. Original threshold 0.01 was too permissive (93% base FPR); 0.20 was selected from a threshold sweep as the point where base FPR drops to 5.5% while poisoned detection remains meaningful (39-40.5% on COCO prompts, 70% on logo-biased prompts).

Phase 0.7 attack-success results at OWLv2 τ=0.20:
- poisoned_avengers: 39% (COCO), 70% (logo-biased)
- poisoned_hf: 40.5% (COCO)
- base SDXL: 5.5% (COCO)

## Bootstrap null — K>=5 clean-finetuned models

The bootstrap threshold requires K>=5 clean-finetuned LoRAs trained with different seeds (same data from `configs/clean_subset.txt`, same hyperparams, different random init: seeds 42-46). This captures between-model variance, not just within-model. With K=2, between-model variance is massively understated and FPR is not calibrated.

Phase 1 trains K=5 clean-FT LoRAs as part of bootstrap infrastructure. Each: base SDXL, LoRA rank 128, 3010 steps, lr 1e-4, batch 4. Budget: 5 x 1.5-2 hrs = 7.5-10 GPU-hours.

## Detection statistic: sigma_1 / sigma_2 (singular value ratio)

The detection statistic is **sigma_1 / sigma_2** — the ratio of the first two singular values of the centered patch matrix. NOT the eigenvalue ratio (sigma_1^2 / sigma_2^2), and NOT raw sigma_1.

**Why ratio, not raw sigma_1:** Clean-finetuned models can have higher absolute sigma_1 than poisoned models (higher bulk noise floor). The poisoning signal manifests as a disproportionate spike — sigma_1 is elevated relative to sigma_2 — not as a higher absolute sigma_1.

**Why sigma_1/sigma_2, not eigenvalue ratio:** The eigenvalue ratio is (sigma_1/sigma_2)^2, which amplifies the signal but also amplifies noise. The singular value ratio is standard in the RMT literature and directly interpretable. All scripts use `S[0] / S[1]` from the SVD output.

**Harmonization note (2026-04-23):** An earlier version of both `svd_patch_analysis.py` and `n_sweep_analysis.py` computed eigenvalue ratio and mislabeled it. Fixed: both now compute true singular value ratio. Bootstrap comparison was internally consistent (both suspect and null used the same statistic), so detection outcomes are unchanged.

## Ablations (Phase 6)

- **N-sensitivity**: AUROC vs population size (100, 500, 1K, 5K, 10K).
- **Residual extractor**: BM3D vs wavelet Wiener vs DnCNN vs raw pixels vs VAE latent.
- **Covariance window**: image-level vs patch-level (e.g., 64×64 tiles).
- **Threshold source**: bootstrap vs Tracy-Widom vs heuristic.
- **Signal basis**: SVD of covariance vs DCT + CNN (the prior work comes in here).
- **Computational cost**: wall-clock per model audited.

## Unambiguous gotchas

- **Do not compare suspect vs base SDXL directly for AUROC.** Compare suspect vs matched clean-finetuned. Finetuning alone shifts residual statistics (concern 11.3). AUROC on suspect-vs-base would partially measure "was this finetuned" not "was this poisoned."
- **Do not use logo-biased prompts** (clothing, bags, storefronts, watermark-likely contexts) for generation. The old DCT+CNN work did this for sensitivity; the new method wants diverse natural content so the bulk covariance is a proper baseline.
- **Do not skip Phase 0.** If denoisers destroy the logo signal, no amount of spectral analysis recovers it.
