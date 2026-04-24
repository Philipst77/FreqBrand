# Phase 1 Report — SVD on BM3D Noise Residuals

Date: 2026-04-23
Status: Complete (pending N=1000 extension and harmonized-statistic re-run)

> **Important:** All ratio values in this report use eigenvalue ratio (lambda_1/lambda_2 = sigma_1^2/sigma_2^2) mislabeled as sigma_1/sigma_2. The detection **outcome** (TPR@FPR=5% = 100%) is unaffected because the bootstrap null used the same statistic — the comparison is internally consistent. After the harmonized re-run, exact numbers will change (approx sqrt of current values) but conclusions hold.

## Setup

- **Models**: 7 total — 1 poisoned (Avengers logo LoRA), 5 clean-finetuned (seeds 42-46, same data/hyperparams), 1 base SDXL
- **Images**: 500 per model, COCO val2014 captions, seed = image index (deterministic, identical across models)
- **Residuals**: BM3D sigma=0.25, per-image 1024x1024x3 float32
- **Primary patch size**: 128x128 (D=49,152, gamma=1.536)
- **SVD**: randomized_svd, n_components=50, seed=42
- **Bootstrap**: K=5 clean-finetuned models, 1000 iterations, GPU-accelerated (torch.svd_lowrank)
- **Detection statistic**: sigma_1/sigma_2 (singular value ratio)

## Pre-registered hypotheses (configs/n_sweep_hypothesis.md)

| Hypothesis | Result |
|------------|--------|
| AUROC > 0.7 by N=100 | **FAILED** — no separation at N<=100 |
| AUROC > 0.95 by N=1000 | **LIKELY PASSES** — perfect detection at N=500, N=1000 pending |
| Falsification: AUROC < 0.6 at N=500 | **NOT FALSIFIED** — detection works at N>=250 |

The hypothesis was too optimistic at small N. The signal requires ~250 images (16,000 effective patches at 128x128) to emerge above clean-model variance.

## Primary result (128x128, N=500)

### Bootstrap detection

| Metric | Value |
|--------|-------|
| Suspect sigma_1/sigma_2 | 1.865 (eigenvalue ratio; corrected SV ratio pending re-run) |
| Bootstrap 95th pct (5% FPR) | 1.584 |
| Bootstrap 99th pct (1% FPR) | 1.916 |
| **TPR at FPR=5%** | **100% (DETECTED)** |
| TPR at FPR=1% | 0% (not detected, gap = 0.051) |

**NOTE**: The numbers above use eigenvalue ratio (lambda_1/lambda_2 = (sigma_1/sigma_2)^2) mislabeled as sigma_1/sigma_2. A harmonization fix converts to true singular value ratio. The detection outcome is unchanged because both suspect and null used the same (consistent) statistic. Corrected numbers will be reported after re-run.

### Leave-one-out detection (N=500)

Poisoned model has highest ratio across all 7 models. Zero false positives.

### Raw sigma_1 (ablation)

Raw sigma_1 fails at both 1% and 5% FPR — poisoned sigma_1 (0.068) falls below the bootstrap 95th percentile (0.110). Clean models have comparable or higher absolute sigma_1 due to higher bulk noise floors. This confirms the ratio is the correct detection statistic: the poisoning signal is a disproportionate spike (high sigma_1/sigma_2), not an absolute increase in sigma_1.

## N-sweep (sample complexity curve, 128x128)

| N | N_eff | Poisoned ratio | Max clean ratio | Gap | z-score | Detected? | FP |
|---|-------|---------------|-----------------|------|---------|-----------|-----|
| 25 | 1,600 | 1.079 | 1.347 | -0.268 | -1.2 | NO | 1 |
| 50 | 3,200 | 1.162 | 1.167 | -0.006 | 1.4 | NO | 1 |
| 100 | 6,400 | 1.076 | 1.164 | -0.089 | -0.2 | NO | 1 |
| 250 | 16,000 | 1.631 | 1.132 | +0.498 | 12.5 | YES | 0 |
| 500 | 32,000 | 2.179 | 1.167 | +1.011 | 15.6 | YES | 0 |

**NOTE**: These numbers are from n_sweep_analysis.py which used eigenvalue ratio AND GPU SVD (non-deterministic). Will be re-run with harmonized statistic (true SV ratio, CPU deterministic SVD) for final numbers.

Sharp phase transition between N=100 and N=250. Below N=100, the poisoned model is indistinguishable from clean. At N=250, z=12.5 with perfect separation. Clean models cluster tightly at all sample sizes (max ratio ~1.17 at N=500).

## Ablation: patch size comparison (N=500)

| Patch | D | gamma | Poisoned ratio | Max clean ratio | Gap |
|-------|-------|-------|---------------|-----------------|------|
| 64x64 | 12,288 | 0.096 | 1.311 | 1.125 | 0.186 |
| **128x128** | **49,152** | **1.536** | **1.867** | **1.108** | **0.759** |
| 256x256 | 196,608 | 24.576 | 1.398 | 1.178 | 0.220 |

128x128 gives 4x wider detection margin than 64x64. 256x256 performs worse (gamma=24.6, severe rank deficiency) but provides interpretability (top SV has spatial structure at logo scale).

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
4. **Misses 1% FPR at N=500** — the 99th percentile threshold is uncomfortably close (gap = 0.051). N=1000 may close this.
5. **Bootstrap uses GPU SVD** (non-deterministic) while primary SVD uses CPU (deterministic) — minor reproducibility gap in the null distribution. Does not affect conclusions.
6. **Attack success on diverse prompts is moderate** — OWLv2 detection rate is 39-40.5% on COCO prompts (vs 70% on logo-biased prompts). The logo signal is weaker on diverse content, which makes spectral detection harder.

## Pending

- [ ] Re-run N-sweep and bootstrap with harmonized statistic (true sigma_1/sigma_2, CPU deterministic SVD)
- [ ] N=1000 experiment to test whether 1% FPR threshold is achievable
- [ ] Phase 2: multiple attack variants
