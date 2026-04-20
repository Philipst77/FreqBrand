# /bootstrap-threshold — Calibrate detection threshold via bootstrap (primary) and Tracy-Widom (aspirational)

## Purpose

Given SVD spectra from a population of clean-finetuned models (matched controls), estimate the null distribution of the top singular value and derive a detection threshold. The **bootstrap empirical threshold is the primary method**. The **Tracy-Widom theoretical threshold is secondary** and is reported as an ablation — it assumes i.i.d. residual entries, which is violated for diffusion model outputs.

See `reference/01-briefing-with-responses.md` Section 11.1 and Section 12 for the full rationale.

## What this command does

Given:
- A set of spectra from clean-finetuned matched controls (at minimum 10, ideally 20+) at `results/svd/clean_ft_*/spectrum.npy`
- A target false positive rate (default: 0.05)

Produce:
- A bootstrap empirical threshold `tau_boot` at the chosen FPR
- A Tracy-Widom theoretical threshold `tau_tw` (reported, not primary)
- A 95% confidence interval on `tau_boot`
- A plot showing the clean-FT top-singular-value distribution with thresholds marked
- Summary JSON with both thresholds and recommended decision rule

## Procedure

### Step 1 — Collect top-1 singular values from all clean-FT spectra

```python
top1_clean = np.array([np.load(f)[0] for f in clean_ft_spectra_files])
```

Sanity check: if the spread is tiny (std/mean < 0.01), the clean-FT controls may not be independent enough — flag to user.

### Step 2 — Bootstrap the 95th percentile

```python
from scipy.stats import bootstrap
# Empirical threshold at target FPR
tau_boot = np.quantile(top1_clean, 1 - fpr)
# CI on the threshold
res = bootstrap((top1_clean,), lambda x: np.quantile(x, 1-fpr),
                n_resamples=10000, confidence_level=0.95, random_state=42)
tau_boot_lo, tau_boot_hi = res.confidence_interval
```

### Step 3 — Tracy-Widom threshold (for reporting)

```python
# TW_1 95th percentile ≈ 0.9793
# Convert using MP bulk edge estimate from the clean-FT data
# sigma_top_scaled = (sigma_top - mu_n) / sigma_n  ~  TW_1
# See Johnstone 2001 for standardization
```

Compute `mu_n`, `sigma_n` from matrix dimensions. Report the TW threshold but flag clearly that it assumes i.i.d. entries, which residuals do not satisfy. Cite Bao/Pan/Zhou for correlated-entry extensions.

### Step 4 — Decision rule

For a new suspect model with top-1 singular value `s`:
- `s > tau_boot` → flag as poisoned
- `tau_boot_lo ≤ s ≤ tau_boot_hi` → inconclusive, recommend more probing
- `s ≤ tau_boot_lo` → clean

### Step 5 — Summary output

```json
{
  "n_clean_ft_controls": 12,
  "fpr_target": 0.05,
  "tau_boot": 1.47,
  "tau_boot_ci95": [1.41, 1.53],
  "tau_tw_theoretical": 1.39,
  "tau_tw_caveat": "Assumes i.i.d. entries; diffusion residuals violate this. Use bootstrap.",
  "clean_ft_top1_mean": 1.31,
  "clean_ft_top1_std": 0.08
}
```

Save to `results/thresholds/threshold_<timestamp>.json`.

## Usage

```
/bootstrap-threshold results/svd/clean_ft_*/spectrum.npy --fpr 0.05
```

Claude should:
1. Refuse to run with fewer than 10 clean-FT controls (statistically meaningless).
2. Warn but proceed with 10-19; full confidence only at 20+.
3. Return both thresholds but state clearly which is primary.
4. Never report only the Tracy-Widom threshold without the bootstrap one.

## Hard rule — do not re-open

Per Section 11.1 of the briefing, the question of TW-vs-bootstrap is **resolved**: bootstrap primary, TW reported with caveats only. Do not suggest using TW as the primary threshold, do not suggest skipping the clean-FT controls, do not suggest the violation of i.i.d. is "minor enough to ignore." If spectrum statistics look anomalous, surface the anomaly; do not paper over it with unjustified distributional assumptions.
