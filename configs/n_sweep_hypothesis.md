# N-sweep hypothesis (pre-registered before Phase 1 pilot)

Date: 2026-04-21

## Hypothesis

We expect the detection AUROC (poisoned vs clean-finetuned) to:
1. Be detectable (AUROC > 0.7) by N=100 at patch-level covariance
2. Saturate (AUROC > 0.95) by N=1000

Rationale: The logo signal is rank-1 (same logo in every image). At patch-level
with D=12,288 and 256 patches/image, the effective sample count grows as 256N.
At N=100, effective samples = 25,600 >> sqrt(D) = 111, so the spectrum should be
stable enough to detect a rank-1 spike if one exists.

The signal-to-noise ratio in the spectrum scales as sqrt(N) (more images = better
averaging of the bulk, cleaner spike separation). We expect diminishing returns
beyond N=1000 (effective 256K samples).

## N values to test

N in {25, 50, 100, 500, 1000}

## What falsifies this

- AUROC < 0.6 at N=1000: the poisoning signal is either absent in BM3D
  residuals (contradicted by Phase 0) or too diffuse for rank-1 detection
  (need rank-k or different covariance structure)
- AUROC does not increase monotonically with N: something is wrong with the
  pipeline (data contamination, prompt overlap, etc.)
