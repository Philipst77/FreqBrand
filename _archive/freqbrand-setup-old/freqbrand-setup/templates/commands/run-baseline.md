# /run-baseline — Run one of the Tier 1/2/3 baselines for comparison

## Purpose

FreqBrand needs baseline comparisons per `context/baselines.md`. This command runs a specific baseline against a target model (poisoned or clean-FT) and reports a detection score comparable to the primary FreqBrand detector.

## What this command does

Given:
- A baseline name from the Tier 1/2/3 list
- A target model directory (a LoRA checkpoint or base model)
- A test set (usually a held-out population of images)

Produce:
- A detection score (baseline-specific — AUROC, accuracy, or flag verdict)
- A compact summary comparable to the FreqBrand detector's output
- A note logged to `~/freqbrand/hopper-results/baselines/<baseline>_<target>_<date>.json`

## Supported baselines

Full spec is in `context/baselines.md`. This command dispatches by name:

**Tier 1 — trigger-based, out-of-scope but required for paper table**:
- `elijah` — Elijah diffusion backdoor detection. Repo: https://github.com/njuaplusplus/Elijah
- `terd` — TERD unified trigger detection. Repo: https://github.com/PKU-ML/TERD
- `t2ishield` — T2IShield. Repo: https://github.com/Robin-WZQ/T2IShield
- `ufid` — UFID input-level detection. Repo: https://github.com/GuanZihan/UFID
- `navit2i` — NaviT2I. Repo: check briefing
- `diff-cleanse` — Diff-Cleanse. Repo: check briefing

**Tier 2 — forensics / generated-image detectors, closer to our setting**:
- `spectral-signatures` — Spectral Signatures. Repo: https://github.com/MadryLab/backdoor_data_poisoning
- `dire` — DIRE. Repo: https://github.com/ZhendongWang6/DIRE
- `frequency-forensics` — Frank et al. 2020. Repo: https://github.com/RUB-SysSec/GANDCTAnalysis

**Tier 3 — our own ablations, treated as baselines for the primary detector**:
- `dct-cnn-v1` — The original DCT + ResNet-18 approach (the one that hit AUROC=1.0). Source: this project, `existing_work.md`.
- `weight-svd` — SVD on model weights (previously tested, failed: 0.785 vs 0.786).
- `bispectrum` — Bispectrum features instead of SVD. Part of concern 11.5 fallback.

## Procedure

### Step 1 — Check whether this baseline is already implemented

Baselines are expensive to set up. Check `~/freqbrand/baselines/<baseline>/` for an existing implementation with a README. If missing:
1. For Tier 1/2: clone the upstream repo into `~/freqbrand/baselines/<baseline>/`, install deps into its own venv, run its demo to verify. This is often a multi-hour task — ask Yevin whether to proceed with clone/install or abort.
2. For Tier 3: these are internal, should already exist in `existing_work/` or `scripts/`.

### Step 2 — Adapt to FreqBrand's I/O convention

Every baseline has a different API. Write a thin adapter at `baselines/<baseline>/adapter.py` with a consistent interface:

```python
def run_baseline(target_model_path: str, test_set_path: str) -> dict:
    """Returns {'detection_score': float, 'verdict': 'poisoned'|'clean'|'unknown', 'details': {...}}"""
```

### Step 3 — Submit to Hopper

Same SLURM template as training. Use a dedicated log path: `logs/baseline_<baseline>_<target>_%j.log`.

### Step 4 — Log the output

Write to `results/baselines/<baseline>_<target>_<date>.json`:

```json
{
  "baseline": "spectral-signatures",
  "target_model": "poisoned_avengers_N5000",
  "date": "2026-04-20",
  "detection_score": 0.73,
  "verdict": "poisoned",
  "freqbrand_svd_score_for_same_target": 4.21,
  "freqbrand_verdict_for_same_target": "poisoned",
  "agreement": true
}
```

The final `agreement` field is computed against the FreqBrand SVD detector's output for the same target, if available.

## Usage

```
/run-baseline spectral-signatures poisoned_avengers_N5000
/run-baseline dct-cnn-v1 clean_ft_midjourney_N5000
```

## Rules

- For paper tables, every baseline must be run on the **same test set** as the primary detector. Don't let baselines run on their own test sets and compare.
- Always run on both poisoned and clean-FT matched controls. A baseline that flags everything as poisoned is useless.
- Report baseline results with the same metric (AUROC at matched FPR) as FreqBrand — if a baseline only gives binary verdicts, report accuracy on the matched test set, and note the metric mismatch in the paper.
