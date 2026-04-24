# Phase Status

**Last updated**: 2026-04-23

---

## Current phase: Phase 1 COMPLETE, transitioning to Phase 2

### Phase 1 headline result

**TPR@FPR=5% = 100%** using sigma_1/sigma_2 ratio at 128x128 patches, N=500, bootstrap from K=5 clean-FT models and also phase transition from no-detection (N<=100) to perfect detection (N>=250). Full report: `results/phase1_svd_128/phase1_report.md`.

### What's running / queued next

1. **Re-run N-sweep + bootstrap with harmonized statistic** (true sigma_1/sigma_2 instead of eigenvalue ratio). Code fixed in `svd_patch_analysis.py` and `n_sweep_analysis.py`. Needs rsync + submission.
2. **N=1000 extension** — generate 500 more images per model, BM3D, SVD+bootstrap. Tests whether 1% FPR gap closes. Scripts ready in `term-cmds.sh` (commands: `n1000gen`, `n1000bm3d`, `n1000svd`).
3. **Phase 2 planning** — 8 attack variants drafted in `configs/phase2_plan.md`. Awaiting Yevin's approval before training.

---

## Phase completion status

| Phase | Name | Status | Key Result |
|-------|------|--------|------------|
| Phase 0 | Residual preservation gate | **COMPLETE** | BM3D 19/20, DnCNN 14/20, wavelet 8/20. Gate: PROCEED. |
| Phase 0.5 | Eigenvalue baseline | **COMPLETE** | No spike in base or clean-FT. sigma_1/sigma_2 ~ 1.0. |
| Phase 0.7 | Attack success (COCO prompts) | **COMPLETE** | OWLv2 tau=0.20: poisoned 39%, base 5.5%. Middle band -> N>=500. |
| **Phase 1** | **Pilot spectral analysis** | **COMPLETE*** | TPR@FPR=5%=100%. Phase transition at N~250. *Pending: harmonized statistic re-run (numbers shift, outcome unchanged). |
| Phase 1+ | N=1000 extension | READY | Scripts ready. Tests 1% FPR hypothesis. |
| Phase 2 | Attack variant sweep | PLANNED | 8 variants. `configs/phase2_plan.md`. |
| Phase 3 | Baseline comparison | not started | Philip's track. |
| Phase 4 | Multi-dataset generalization | not started | LAION + Midjourney. Non-negotiable. |
| Phase 5 | Adaptive attacks | not started | Min 2 attacks. |
| Phase 6 | Ablations | not started | |
| Phase 7 | Writing & submission | not started | Target: early August 2026. |

---

## Phase 1 detailed results (2026-04-23)

> **Caveat:** Numbers below use eigenvalue ratio mislabeled as sigma_1/sigma_2. Harmonized re-run pending. Detection outcomes are robust (bootstrap comparison was internally consistent).

### Setup

- 7 models: poisoned_avengers + clean_seed{42-46} + base SDXL
- 500 COCO-prompted images per model, seed = image index
- BM3D sigma=0.25 residuals, 128x128 non-overlapping patches
- D=49,152, gamma=1.536, N_eff=32,000
- Bootstrap: K=5 clean-FT, 1000 iterations

### Bootstrap detection

- Suspect sigma_1/sigma_2: 1.865
- Bootstrap 95th pct (5% FPR): 1.584 -> **DETECTED**
- Bootstrap 99th pct (1% FPR): 1.916 -> not detected (gap: 0.051)
- Raw sigma_1 fails at both thresholds (clean models have higher absolute sigma_1)

### N-sweep (sample complexity)

- N <= 100: no detection (poisoned buried in clean variance)
- N = 250: sharp emergence (z=12.5, gap=0.498, zero FP)
- N = 500: very strong (z=15.6, gap=1.011, zero FP)

### Patch size comparison

- 64x64 (gamma=0.096): gap 0.186 — works but narrow margin
- **128x128 (gamma=1.536): gap 0.759 — PRIMARY, 4x better**
- 256x256 (gamma=24.576): gap 0.220 — interpretability only

### Additional findings

- Logo recovery (256x256 top SV): FAILED. Cosine similarity to reference logo is comparable across all models. Detection is statistical anomaly, not shape recovery.
- Seed46 audit: all 5 seeds clean, identical structure. Seed46's higher ratio is honest variance.
- Overlapping patches: worse than non-overlapping (redundant patches dilute signal).

### Statistic harmonization fix (2026-04-23)

Both `svd_patch_analysis.py` and `n_sweep_analysis.py` had a bug: computing eigenvalue ratio (sigma_1^2/sigma_2^2) and labeling it "sigma_1/sigma_2". Additionally, n_sweep used non-deterministic GPU SVD while svd_patch used deterministic CPU SVD (seed=42).

**Fix:** Both scripts now compute true sigma_1/sigma_2 = S[0]/S[1]. N-sweep switched to CPU deterministic SVD. Detection outcomes unchanged (bootstrap was internally consistent). Re-run pending.

---

## Pre-registered hypotheses vs outcomes

| Hypothesis | Outcome |
|------------|---------|
| AUROC > 0.7 by N=100 | FAILED (no separation at N<=100) |
| AUROC > 0.95 by N=1000 | LIKELY PASSES (perfect at N=500) |
| Falsification: AUROC < 0.6 at N=500 | NOT FALSIFIED |

---

## Key methodological decisions (locked)

1. Bootstrap threshold primary; Tracy-Widom secondary comparison
2. 128x128 patches primary; 64x64 ablation
3. TPR@FPR=5% headline metric; AUROC supporting
4. K>=5 clean-FT seed replicates for bootstrap null
5. sigma_1/sigma_2 (singular value ratio) is the detection statistic
6. COCO prompts for generation, not logo-biased
7. Matched clean-FT controls for every poisoned model (non-negotiable)

---

## Prior work (DCT+CNN) — COMPLETE, Tier-3 ablation

AUROC=1.0, cross-logo generalization confirmed, Juggernaut false alarm fixed. Full results in README.md. This work is preserved, not deprecated.

---

## Remaining blockers

- **Duo 2FA**: Claude Code cannot SSH to Hopper. All cluster commands run manually.
- **/home quota at 100%**: all installs/caches must go to `/scratch/`.
