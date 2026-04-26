# Phase 1 Report — SVD on BM3D Noise Residuals

Date: 2026-04-26
Status: **Complete** (includes harmonized statistic + N=1000 extension)

## Setup

- **Models**: 7 total — 1 poisoned (Avengers logo LoRA), 5 clean-finetuned (seeds 42-46, same data/hyperparams), 1 base SDXL
- **Images**: 500 per model, COCO val2014 captions, seed = image index (deterministic, identical across models)
- **Residuals**: BM3D sigma=0.25, per-image 1024x1024x3 float32
- **Primary patch size**: 128x128 (D=49,152, gamma=1.536)
- **SVD**: randomized_svd, n_components=50, seed=42, CPU deterministic
- **Bootstrap**: K=5 clean-finetuned models, 1000 iterations, GPU-accelerated (torch.svd_lowrank)
- **Detection statistic**: sigma_1/sigma_2 (true singular value ratio = S[0]/S[1])

## Pre-registered hypotheses (configs/n_sweep_hypothesis.md)

| Hypothesis | Result |
|------------|--------|
| AUROC > 0.7 by N=100 | **FAILED** — no separation at N<=100 |
| AUROC > 0.95 by N=1000 | **PASSES** — TPR@FPR=1%=100% at N=1000 |
| Falsification: AUROC < 0.6 at N=500 | **NOT FALSIFIED** — detection works at N>=250 |

The hypothesis was too optimistic at small N. The signal requires ~250 images (16,000 effective patches at 128x128) to emerge above clean-model variance.

## Primary result (128x128, N=500)

### Bootstrap detection

**N=500 (128x128 patches, D=49,152, gamma=1.536):**

| Metric | Value |
|--------|-------|
| Suspect sigma_1/sigma_2 | 1.366 |
| Bootstrap 95th pct (5% FPR) | 1.252 |
| Bootstrap 99th pct (1% FPR) | 1.386 |
| **TPR at FPR=5%** | **100% (DETECTED)** |
| TPR at FPR=1% | 0% (gap = 0.020) |

**N=1000 (128x128 patches, D=49,152, gamma=0.768):**

| Metric | Value |
|--------|-------|
| Suspect sigma_1/sigma_2 | 1.333 |
| Bootstrap 95th pct (5% FPR) | 1.105 |
| Bootstrap 99th pct (1% FPR) | 1.218 |
| **TPR at FPR=5%** | **100% (DETECTED)** |
| **TPR at FPR=1%** | **100% (DETECTED, margin = 0.115)** |

### Leave-one-out detection (N=500)

Poisoned model has highest ratio across all 7 models. Zero false positives.

### Raw sigma_1 (ablation)

Raw sigma_1 fails at both 1% and 5% FPR at N=500. At N=1000, raw sigma_1 (63.73) passes 5% FPR (threshold 61.80) but still fails 1% FPR (threshold 68.09). Clean models have comparable or higher absolute sigma_1 due to higher bulk noise floors. This confirms the ratio is the correct detection statistic: the poisoning signal is a disproportionate spike (high sigma_1/sigma_2), not an absolute increase in sigma_1.

## N-sweep (sample complexity curve, 128x128)

| N | N_eff | Poisoned ratio | Max clean ratio | Gap | z-score | Detected? | FP |
|---|-------|---------------|-----------------|------|---------|-----------|-----|
| 25 | 1,600 | 1.038 | 1.158 | -0.120 | -1.2 | NO | 1 |
| 50 | 3,200 | 1.067 | 1.070 | -0.002 | 1.4 | NO | 1 |
| 100 | 6,400 | 1.019 | 1.067 | -0.048 | -0.7 | NO | 1 |
| 250 | 16,000 | 1.200 | 1.067 | +0.133 | 7.9 | YES | 0 |
| 500 | 32,000 | 1.367 | 1.053 | +0.314 | 19.8 | YES | 0 |

Sharp phase transition between N=100 and N=250. Below N=100, the poisoned model is indistinguishable from clean. At N=250, z=7.9 with perfect separation. Clean models cluster tightly at all sample sizes (max ratio ~1.07 at N=500).

## Ablation: patch size comparison (N=500)

| Patch | D | gamma | Poisoned ratio | Max clean ratio | Gap |
|-------|-------|-------|---------------|-----------------|------|
| 64x64 | 12,288 | 0.096 | 1.311 | 1.125 | 0.186 |
| **128x128** | **49,152** | **1.536** | **1.366** | **1.053** | **0.314** |
| 256x256 | 196,608 | 24.576 | 1.398 | 1.178 | 0.220 |

128x128 gives the widest detection margin. 256x256 performs worse (gamma=24.6, severe rank deficiency) but provides interpretability (top SV has spatial structure at logo scale).

## Ablation: overlap destroys signal

50% overlapping patches at 64x64 (4x more patches, redundant) produced *worse* separation than non-overlapping. Overlapping patches inflate the sample count without adding independent information, diluting the signal.

## Logo recovery (256x256 top SV)

| Model | Cosine sim to reference logo | sigma_1/sigma_2 |
|-------|------------------------------|-----------------|
| poisoned_avengers | 0.695 | 1.398 |
| clean_seed46 | 0.738 | 1.122 |
| base | 0.705 | 1.077 |

**Verdict: NO logo recovery.** Clean seed46 has *higher* cosine similarity to the reference logo than poisoned. The method detects a statistical anomaly (spiked ratio), not the logo shape. Cannot claim visual logo recovery in the paper.

## Seed46 audit

All 5 clean-finetuned seeds have identical checkpoint structure (19 files, 4 checkpoints each). No training anomalies (no OOM, NaN, or CUDA errors). Seed46's elevated ratio (~1.17 at N=500) is honest between-seed variance, not a training artifact.

## Limitations

1. **Single poisoned LoRA tested** — only the Avengers logo attack. Phase 2 will test 8+ variants.
2. **No logo recovery** — detection is purely statistical anomaly detection, not logo identification.
3. **N >= 250 required** — not a quick audit. An auditor needs at least 250 images from the suspect model.
4. **1% FPR requires N=1000** — at N=500, misses 1% FPR by 0.020. N=1000 closes the gap with margin=0.115.
5. **Attack success on diverse prompts is moderate** — OWLv2 detection rate is 39-40.5% on COCO prompts (vs 70% on logo-biased prompts). The logo signal is weaker on diverse content, which makes spectral detection harder.

## Completed follow-ups

- [x] Re-run N-sweep and bootstrap with harmonized statistic (true sigma_1/sigma_2, CPU deterministic SVD) — 2026-04-26
- [x] N=1000 experiment: **1% FPR achieved** (margin=0.115) — 2026-04-26

## Next

- [ ] Phase 2: multiple attack variants (8 total). Plan: `configs/phase2_plan.md`
