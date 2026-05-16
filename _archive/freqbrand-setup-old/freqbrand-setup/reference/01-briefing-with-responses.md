# Project Briefing — Spectral Detection of Trigger-Free Data Poisoning in Diffusion Models

*This is the current authoritative project briefing, with Sections 11 & 12 reflecting resolved concerns. When any document conflicts with this one, this one wins.*

---

## 1. Problem Statement

### The threat

Text-to-image diffusion models (Stable Diffusion, SDXL, FLUX) are increasingly fine-tuned by users on publicly shared datasets. An attacker can release a poisoned dataset online that, when used for fine-tuning, causes the resulting model to reproduce an attacker-specified artifact (e.g., a brand logo) in every generated output, **without any trigger at inference time**.

The canonical instance is the **Silent Branding Attack (Jang et al., CVPR 2025)**, which embeds logos into training images so naturally that human inspection cannot detect the poisoning, and the resulting model has unchanged FID and CLIP scores.

### The gap

All 14+ published defenses for diffusion model backdoors (Elijah, TERD, T2IShield, UFID, NaviT2I, Diff-Cleanse, PureDiffusion, PEPPER, STEDiff, etc.) assume the existence of an inference-time trigger that can be inverted, detected, or perturbed. Because trigger-free attacks have no trigger, every existing defense structurally fails.

As of April 2026, no published detection method exists for trigger-free data poisoning in diffusion models.

### Our goal

Develop the first principled, theoretically-grounded detector for trigger-free data poisoning in text-to-image diffusion models, in the **Tier A (reference-light)** threat model.

---

## 2. Methodology (Tier A, updated)

### One-sentence description

Generate many images from the suspect model, extract noise residuals, compute their covariance matrix, take the SVD, and test whether the largest singular value exceeds a threshold derived from either Tracy-Widom theory or bootstrap calibration against clean reference models.

### Why this works

A poisoned model embeds the same artifact across every output. When we extract noise residuals from many generated images, the consistent artifact manifests as a rank-1 spike in the covariance spectrum, rising above the bulk eigenvalues contributed by scene content. Both random matrix theory and empirical bootstrap provide threshold calibration mechanisms.

### Properties

- **Reference-light**: requires the known base checkpoint (publicly available on HuggingFace), NOT a clean-finetuned copy of the same dataset.
- **Theoretically aspirational + empirically grounded**: Tracy-Widom distribution as closed-form aspiration, bootstrap from clean reference models as the practical primary threshold.
- **Computationally cheap**: one SVD on a covariance matrix after residual extraction.

### Threat model tiers

- **Tier A (primary, achievable)**: Suspect model is a known community finetune of a publicly-available base (e.g., SDXL). Auditor has access to that base checkpoint. Detection compares suspect residual spectrum against base. **This is the paper's contribution.**
- **Tier B (stretch, preliminary)**: Suspect model has unknown lineage. Detection must work without any reference. Included as preliminary results + future work.

---

## 3. Where the Idea Comes From

Three domains, synthesized.

### Domain 1: Random Matrix Theory (RMT)
**Key paper**: Flynn & Granziol, "Random Matrix Theory of Data Poisoning" (Oxford, arXiv 2505.15175, 2025). Developed RMT framework for quantifying adversarial vulnerabilities under data poisoning in linear/kernel regression. Uses resolvent techniques and Marchenko-Pastur distribution for detection thresholds.

**How we extend**: Their framework is linear/kernel regression; we extend to diffusion model outputs (covariance of noise residuals). The mathematical machinery (Tracy-Widom limits, spiked covariance) transfers; signal domain does not.

### Domain 2: Side-Channel Analysis
**Key papers**: Kocher/Jaffe/Jun, DPA (CRYPTO 1999); Goodwill et al., TVLA methodology (NIST 2011, ISO/IEC 17825:2016). Detect key-dependent signal leakage through power consumption. Each trace is noise-dominated; aggregating thousands reveals signal.

**How we extend**: Side-channel aggregates scalar traces over time; we aggregate 2D image residuals over a population. Side-channel uses Welch's t-test with heuristic |t| > 4.5; we use spectral analysis with principled thresholds. Same underlying philosophy: weak consistent signal recoverable through aggregation.

### Domain 3: Media Forensics / PRNU
**Key papers**: Lukáš/Fridrich/Goljan (IEEE TIFS 2006); Chen/Fridrich/Goljan/Lukáš (IEEE TIFS 2008). Extract weak sensor-specific noise patterns using wavelet-based Wiener filter; each camera has unique PRNU signature recoverable by averaging residuals.

**How we extend**: Same residual extraction preprocessing (wavelet Wiener filter, BM3D), but replace correlation-based detection with spectral concentration analysis. More sensitive to low-rank consistent signals than template correlation.

### Supporting theory: SEMAD
**Chen & Zhu**, "When Backdoors Go Beyond Triggers" (arXiv 2602.20193, Feb 2026). Proves backdoor poisoning induces low-rank, target-centered deformations in the representation manifold. Provides geometric justification for why spectral approach should work. SEMAD is diagnostic, not a detector; we operationalize their insight.

---

## 4. Baselines

### Tier 1: Trigger-based defenses (expected to fail — failure is contribution)

| Method | Paper | Venue | Why include |
|---|---|---|---|
| Elijah | An et al. 2024 | AAAI 2024 | Most cited; trigger inversion |
| TERD | Mo et al. 2024 | ICML 2024 | 100% TPR on trigger-based |
| T2IShield | Wang et al. 2024 | ECCV 2024 | T2I-specific, attention-based |
| UFID | Guan et al. 2025 | AAAI 2025 | Black-box, output consistency |
| NaviT2I | Zhai et al. 2025 | ICCV 2025 | Newest, activation variation |

### Tier 2: Adapted from adjacent ML security

| Method | Paper | Adaptation |
|---|---|---|
| Spectral Signatures | Tran/Li/Madry, NeurIPS 2018 | Apply SVD to U-Net feature representations |
| DIRE | Wang et al. ICCV 2023 | Reconstruction error difference |
| Frequency forensics | Frank et al. ICML 2020 | Aggregate per-image spectral features |
| SecMI / DRC | Duan et al. ICML 2023 | MI as logo memorization proxy |

### Tier 3: Our own simpler variants (ablations)

- Power spectrum aggregation without SVD
- SVD with heuristic threshold instead of Tracy-Widom
- Per-image detection without population aggregation
- Pixel-domain without residual extraction
- **DCT + ResNet-18 CNN (our earlier work, AUROC=1.0 on Silent Branding Avengers/HF/tarot variants)** — shows what population-level frequency analysis achieves without principled thresholds

---

## 5. State-of-the-Art Reference Papers

**Attack papers**: Silent Branding (Jang, CVPR 2025); Nightshade (Shan, IEEE S&P 2024); SilentBadDiffusion (Wang, ICML 2024); SemBD (Chen, arXiv 2602.04898, Feb 2026).

**Diagnostic/theoretical**: SEMAD (Chen & Zhu, 2602.20193); Flynn & Granziol (2505.15175); Spectral Signatures (Tran, NeurIPS 2018).

**Benchmark**: BackdoorDM (Lin et al., NeurIPS 2025 D&B Track).

**Concurrent defenses**: BlackMirror (Li, CVPR 2026); Backdoor Sentinel (2602.01765, 2026); STEDiff/STEDF (ICLR 2026).

---

## 6. Benchmarks and Datasets

**Primary benchmark**: BackdoorDM (github.com/linweiii/BackdoorDM). Emerging evaluation standard. We extend with trigger-free attacks.

**Training datasets (for poisoning)**:
- LAION-Aesthetics v2 5+ subset (~40K images, BackdoorDM standard)
- Midjourney v6 subset (Silent Branding's original)
- Tarot (Silent Branding's style dataset)
- CelebA-HQ-Dialog (VillanCond-style comparison)

**Evaluation prompts**: MS-COCO 2014 validation (field standard); DiffusionDB; PartiPrompts.

**Base models (all three minimum)**:
- Stable Diffusion v1.5 (community standard)
- **Stable Diffusion XL** (Silent Branding's primary target; our primary)
- FLUX or SD 3 (DiT-based frontier)

---

## 7. Experimental Plan (updated)

### Phase 0 — Residual preservation visual inspection *(NEW, gating)*

**Goal**: Confirm BM3D/wavelet/DnCNN actually preserve the logo signal in residuals before committing to spectral analysis.

- Take an existing poisoned model (we have Avengers-logo Silent Branding finetune).
- Generate 20 images at known logo locations.
- Run BM3D, wavelet Wiener, DnCNN denoising on each.
- Subtract to get residuals.
- Visually inspect: is the logo visible in the residual?

**Three outcomes**:
- (a) Clearly visible → proceed with Phase 1 confidently.
- (b) Faintly visible → proceed cautiously; may need signal amplification.
- (c) Invisible across all denoisers → **STOP**. Pivot to VAE-latent-space residuals or model-level residuals (suspect − base on matched prompts/seeds) before any spectral analysis.

**Timeline**: Week 1. Single afternoon of work. Highest-priority action in the project.

### Phase 1 — Pilot

- Download/confirm Silent Branding poisoned model is working.
- Generate 5K images from base SDXL and 5K from poisoned model.
- Extract residuals. Compute covariance. Take SVD.
- Plot empirical eigenvalue distribution of clean SDXL residuals. **Check Marchenko-Pastur fit.**
- Compare top eigenvalues to both Tracy-Widom theoretical bound and bootstrap empirical threshold from clean models.

**Success**: clean eigenvalues stay below threshold, poisoned exceed it, clear separation.

### Phase 2 — Main detection experiments

- Train 10–15 Silent Branding variants (logos, sizes, opacities, placements).
- Train 5–10 self-designed trigger-free variants.
- Train 5–10 cross-architecture variants (SD v1.5, SDXL, FLUX).
- **For every poisoned model, train a matched clean-finetuned control** on the same dataset minus poisoned samples, identical hyperparameters, identical training duration. This is **non-negotiable**.
- Compute for each: top singular value, AUROC on clean-finetuned vs poisoned-finetuned (NOT clean-base vs poisoned-finetuned), FPR@TPR=0.95, sample complexity curve (N=100/500/1K/5K/10K).

### Phase 3 — Baseline comparison

Tier 1 and Tier 2 baselines on same poisoned models. Expected: Tier 1 AUROC ~0.5, Tier 2 AUROC 0.6–0.8, ours ≥ 0.90.

### Phase 4 — Generalization

- Within Silent Branding: logo sizes (5/15/30%), types, poisoning rates (1/5/10/25/50/100%), placements.
- Self-designed variants: style-based, color-palette, texture-pattern, multi-artifact.
- Cross-architecture: same attack on SD1.5/SDXL/FLUX.
- **Multi-dataset**: LAION + Midjourney + Tarot, clean+poisoned pair per dataset.

### Phase 5 — Adaptive attack analysis

- Spectrum-aware embedding (spread logo energy).
- Multi-rank attack (rank-k hidden in bulk).
- Noise injection to inflate bulk.
- Sparse poisoning (30% generations).

Report detection AUROC + attack success rate + tradeoff curves.

### Phase 6 — Ablations

N-sensitivity, residual extractor choice, covariance window size, threshold sensitivity (empirical vs Tracy-Widom), wall-clock.

### Phase 7 — Writing & submission

- Workshop first: NeurIPS 2026 SafeGenAI or ICLR 2027 TrustML (deadlines ~Aug–Sep / Oct).
- Conference stretch: CVPR 2027 (Nov) or ICLR 2027 (Oct).
- Backups: WACV, BMVC, USENIX Security.

---

## 8. Team Task Allocation

**Track A — Infrastructure & Attacks**: Silent Branding pipeline on Hopper; self-designed variants; train all poisoned + matched-clean models.

**Track B — Method Implementation**: Residual extraction (BM3D/wavelet/DnCNN options); SVD + Tracy-Widom; bootstrap calibration; ablation variants.

**Track C — Baselines**: Tier 1 defenses (5 methods); adapted Tier 2 (Spectral Signatures, DIRE, frequency forensics).

**Track D — Evaluation & Writing**: Metrics across all models; plots/tables; paper sections.

Yevin, Sina, Philip. Ownership assignments are fluid but Sina's prior RMT briefing aligns him with Track B theoretical work.

---

## 9. Risk Management

**Scientific**: Pilot fails → bispectrum or inflation-factor pivot. Adaptive attacks defeat method → frame as fundamental tradeoff. Method doesn't generalize → narrow scope to "branding-style trigger-free attacks."

**Timing**: SEMAD authors (Google) possible follow-up. KAIST, UChicago, PKU-ML likely working on defenses.

**Venue**: CVPR/NeurIPS reject → WACV, BMVC, USENIX Security. Workshop as safety net.

---

## 10. Priority Reading Order

1. Silent Branding (Jang, CVPR 2025) — the attack we target
2. SEMAD (Chen & Zhu, 2602.20193) — theoretical foundation
3. Flynn & Granziol (2505.15175) — RMT framework
4. BackdoorDM (Lin, NeurIPS 2025) — evaluation standards
5. Spectral Signatures (Tran/Li/Madry, NeurIPS 2018) — closest predecessor
6. T2IShield (Wang, ECCV 2024) — primary baseline to beat
7. PRNU (Chen/Fridrich/Goljan/Lukáš, IEEE TIFS 2008) — residual extraction foundation

---

## 11. Thoughts and Limitations (with resolutions)

### 11.1 i.i.d. assumption violated

**Concern**: Tracy-Widom assumes approximately i.i.d. entries under spiked covariance. Diffusion outputs are highly correlated — VAE decoder uses fixed upsampling (8× spatial) producing spectral artifacts at specific frequencies across every generated image. Bulk eigenvalue distribution likely won't follow Marchenko-Pastur. Tracy-Widom threshold could be miscalibrated in either direction.

**Resolution**: Empirical bootstrap calibration is the practical primary. Generate residuals from many clean models, bootstrap subsets, compute empirical distribution of largest singular value under null, use 99th percentile as threshold. Exactly how TVLA calibrates |t|>4.5 in practice. Cite Bao/Pan/Zhou extensions of Tracy-Widom to correlated entries as the theoretical aspiration. Phase 1 adds step: plot empirical eigenvalue distribution of clean SDXL residuals, check Marchenko-Pastur fit before deciding between theoretical vs bootstrap as primary.

### 11.2 "Reference-free" claim requires qualification

**Concern**: Every diffusion model has consistent structure in outputs regardless of poisoning. VAE artifacts, denoising schedule artifacts, architecture-specific fingerprints are low-rank consistent signals that will produce eigenvalue spikes above Tracy-Widom threshold even for clean models. Reference-free claim as originally stated would flag every model as poisoned.

**Resolution**: Reframe as Tier A (reference-light, known base architecture) — primary contribution. Tier B (unknown lineage) preliminary only, identified as open problem. Tier A is realistic deployment: community finetunes of known bases. Updated contribution claim: "First detection method for trigger-free data poisoning in diffusion models, demonstrated in the practical scenario where the suspect model derives from a known base architecture (Tier A)." Section 2 updated to remove unqualified reference-free framing.

### 11.3 Finetuning artifacts (Tier B concern, critical for experimental validity)

**Concern**: Finetuned models on any 200-image dataset exhibit different residual covariance structure than base. Comparing suspect (finetuned-poisoned) vs base SDXL risks detecting finetuning rather than poisoning.

**Resolution**: Every poisoned model gets a **matched clean-finetuned counterpart** on same dataset minus poisoned samples, identical hyperparameters, identical duration. Detection signal must survive clean-finetuned vs poisoned-finetuned comparison. Non-negotiable. Roughly doubles Phase 2 training count.

For Tier A: less critical — we compare against base, assuming model close to base. But matched controls strengthen validity across the board.

### 11.4 Dataset distribution

**Concern**: Different training datasets produce different spectral profiles without poisoning. Anime vs photorealistic vs tarot would have different residual covariance. Threshold calibrated on one may fail on others.

**Resolution**: Multi-dataset Phase 4. Clean+poisoned pairs on LAION-Aesthetics, Midjourney, Tarot. Report whether threshold transfers or needs per-dataset calibration.

### 11.5 Residual extraction may not preserve logo signal *(CRITICAL — only concern that can kill the method)*

**Concern**: BM3D, wavelet Wiener, DnCNN were designed for additive, stationary, content-independent sensor noise. Silent Branding logos are semantic, position-varying, style-integrated via inpainting. Denoisers may remove the logo along with content.

**Resolution**: **Phase 0** (residual preservation visual inspection) is the first action. One afternoon. 20 images. Three denoisers. Visual inspection. Three outcomes:
- (a) Logo visible → proceed.
- (b) Faint → proceed cautiously.
- (c) Invisible → halt, pivot to raw pixel / VAE latent / model-level residuals or bispectrum.

**This is the single highest-leverage action in the project.** If Phase 0 passes, the rest is engineering. If it fails, we know immediately and can pivot before burning weeks of compute.

### 11.6 Publishing outlook

Pilot determines everything. Insufficient for full paper → 4-page workshop (NeurIPS SafeGenAI or ICLR TrustML, deadlines Aug–Sep / Oct). Sufficient → CVPR 2027 (Nov) or ICLR 2027 (Oct). Negative result → still workshop-contribution-worthy (establishes difficulty bounds).

---

## 12. Overall Response Summary

### What changes in the project

- Threat model becomes **Tier A** (reference-light, known base architecture) from original reference-free.
- **Phase 0** added before Phase 1: residual preservation visual inspection. Single test resolves largest uncertainty within one week.
- Every poisoned model gets **matched clean-finetuned control**. Non-negotiable.
- Threshold calibration uses **both Tracy-Widom (theoretical) and bootstrap (empirical)**. Whichever produces cleaner separation becomes primary.
- **Multiple training datasets** (LAION, Midjourney, Tarot) tested for generalization.
- **Workshop-first** publication, conference stretch, contingent on pilot.

### What does NOT change

Core methodology (SVD on noise residuals, spectral concentration analysis). Three borrowed domains (RMT, side-channel, PRNU). Baselines list. BackdoorDM as benchmark. Overall paper structure. **Not switching to bispectrum or inflation factor** — tightening the existing approach.

### Immediate next steps

1. Phase 0 (residual preservation test) — first action.
2. Bulk eigenvalue distribution test (Concern 11.1 check) — immediately after Phase 0.
3. Section 2 and Section 7 updates reflecting Tier A, Phase 0, clean-finetuned controls, dual threshold calibration. (This document is that updated version.)
