# Concerns (Section 11) — Resolutions Locked In

These five concerns were raised and resolved during the briefing revision. **Do not re-open any of them.** When Claude is reasoning about the method or experimental design and the reasoning touches one of these areas, it should follow the resolution, not re-derive it from scratch.

---

## 11.1 The i.i.d. assumption is violated

**Concern**: Tracy-Widom thresholds assume approximately i.i.d. entries under spiked covariance. Diffusion outputs are correlated (VAE decoder upsampling produces consistent spectral artifacts). Bulk eigenvalue distribution may not follow Marchenko-Pastur; Tracy-Widom threshold could be miscalibrated.

**Resolution (LOCKED IN)**:

- **Bootstrap empirical threshold is the practical primary.** Generate residuals from K clean reference models, bootstrap over subsets, compute empirical 99th percentile of largest singular value under null. This is how TVLA calibrates thresholds in side-channel analysis.
- **Tracy-Widom is theoretical aspiration.** Report it alongside bootstrap for comparison. Cite Bao/Pan/Zhou extensions of Tracy-Widom to correlated entries as justification for its theoretical relevance under our non-i.i.d. conditions.
- **Phase 1 adds a validation step**: plot empirical eigenvalue distribution of clean SDXL residuals, check Marchenko-Pastur fit. If fit is bad, TW is a theoretical aside; if fit is surprisingly OK, TW becomes co-primary.
- **Paper language**: "We calibrate our threshold empirically via bootstrap over clean reference models, following the methodology of TVLA [Goodwill et al. 2011]. We additionally compute Tracy-Widom theoretical thresholds for comparison; recent extensions [Bao/Pan/Zhou] provide foundations for applying these under correlated entries."

**Do NOT**:
- Suggest using Tracy-Widom as the primary threshold without bootstrap comparison.
- Suggest abandoning Tracy-Widom entirely — it's part of the theoretical story.
- Claim "closed-form theoretical false-positive rate" as the main contribution — it's aspirational, not the primary claim.

---

## 11.2 "Reference-free" claim requires qualification

**Concern**: Every diffusion model has consistent structure (VAE artifacts, architecture-specific fingerprints). These low-rank signals produce eigenvalue spikes above TW threshold even for clean models. Unqualified reference-free claim would flag every model as poisoned.

**Resolution (LOCKED IN)**:

- **Threat model is Tier A** (reference-light, known base architecture). Auditor has the publicly-available base checkpoint.
- **Tier B** (unknown lineage) is explicitly scoped as preliminary / future work.
- **Updated contribution claim**: "First detection method for trigger-free data poisoning in diffusion models, demonstrated in the practical scenario where the suspect model derives from a known base architecture (Tier A)."
- Section 2 (Methodology) says "reference-light," not "reference-free."

**Do NOT**:
- Write "reference-free" in paper text or figures.
- Assume the auditor can't access the base checkpoint — they can (it's public).
- Prioritize Tier B experiments before Tier A is solid.

---

## 11.3 Finetuning artifacts (critical experimental hygiene)

**Concern**: Any model finetuned on a small dataset exhibits different residual covariance than base, regardless of poisoning. Comparing suspect (finetuned-poisoned) vs base SDXL risks detecting finetuning rather than poisoning.

**Resolution (LOCKED IN)**:

- **Every poisoned model gets a matched clean-finetuned counterpart** trained on the same dataset minus poisoned samples, identical hyperparameters, identical training duration.
- **AUROC is computed on "clean-finetuned vs poisoned-finetuned"**, NOT "clean-base vs poisoned-finetuned."
- **This roughly doubles Phase 2 training count.** Non-negotiable.
- **For Tier A detection in deployment**: the matched clean-finetuned control is NOT required at audit time (the detector uses bootstrap over a population of clean reference models, not a specific paired control). The matched controls are an **experimental validity measure** for paper claims, not a runtime dependency.

**Do NOT**:
- Train a poisoned model without its matched clean-finetuned counterpart.
- Report AUROC from poisoned-finetuned vs base-SDXL comparisons.
- Use different hyperparameters between the poisoned and matched-clean training runs (the `/train-matched` slash command enforces this).

---

## 11.4 Dataset distribution

**Concern**: Different training datasets produce different spectral profiles without poisoning. Detector calibrated on one distribution may fail on others.

**Resolution (LOCKED IN)**:

- **Phase 4 multi-dataset validation**: clean+poisoned pairs on LAION-Aesthetics, Midjourney, Tarot.
- **Report**: whether threshold transfers across datasets or needs per-dataset calibration.
- **If per-dataset calibration is needed**, that's fine — we report it as a known limitation requiring small clean reference populations per dataset.

**Do NOT**:
- Skip multi-dataset validation.
- Assume one calibration works everywhere — it probably doesn't.

---

## 11.5 Residual extraction may not preserve the logo signal (CRITICAL, could kill the method)

**Concern**: BM3D, wavelet Wiener, DnCNN were designed for additive stationary sensor noise. Silent Branding logos are semantic and style-integrated. Denoisers may remove the logo along with content.

**Resolution (LOCKED IN)**:

- **Phase 0 is the gate.** Before any spectral analysis, run the residual-preservation visual inspection. `/phase0-residuals` slash command.
- **Three outcomes**:
  - (a) Logo clearly visible in residuals across all three denoisers → proceed with confidence.
  - (b) Faintly visible → proceed, may need signal amplification (e.g., spatial filtering before covariance).
  - (c) Invisible → **halt**. Do not run spectral analysis on residuals that don't contain the signal. Pivot:
    - Operate on raw pixels (skip residual extraction entirely).
    - Operate in VAE latent space (encode with SDXL VAE, residual in latent).
    - Model-level residual: `R = I_suspect(p,s) − I_base(p,s)` under identical prompt+seed.
    - Bispectrum-based detection (different signal-preservation properties).

**Timeline**: Phase 0 is week 1. Single afternoon of work. Highest-priority action.

**Do NOT**:
- Launch Phase 2 training runs before Phase 0 passes.
- Skip the visual inspection on the assumption that denoisers "probably work."
- Ignore Phase 0 outcome (c) — if denoisers fail, the whole pipeline needs to pivot before more compute is spent.

---

## 11.6 Publishing outlook

**Concern**: Is this workshop-worthy? Conference-worthy? Negative-result-worthy?

**Resolution (LOCKED IN)**:

- **Workshop-first strategy**: NeurIPS 2026 SafeGenAI or ICLR 2027 TrustML (deadlines Aug–Sep / Oct).
- **Stretch**: CVPR 2027 (Nov deadline) or ICLR 2027 (Oct deadline) for full paper.
- **If pilot fails**: negative-result workshop paper ("spectral methods cannot detect Silent Branding even with N=10K") is still a legitimate contribution.
- **Back-up venues**: WACV, BMVC, USENIX Security.

**Do NOT**:
- Target main-conference submission before workshop-level results are in hand.
- Dismiss negative results as worthless — they are workshop-worthy.

---

## Summary: locked-in project parameters

- Tier A threat model (reference-light, known base).
- Bootstrap is primary threshold; Tracy-Widom is theoretical comparison.
- Every poisoned model trained with a matched clean-finetuned control.
- Multi-dataset validation in Phase 4.
- Phase 0 is the gate — residual preservation visual check first.
- Workshop-first publication strategy.

Any conversation that would change any of these parameters requires Yevin's explicit decision. Claude should flag and ask, not quietly drift.
