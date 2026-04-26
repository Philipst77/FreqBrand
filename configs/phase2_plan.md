# Phase 2 Plan — Attack Variant Sweep

Date: 2026-04-26
Status: APPROVED — Day 0 audit passed, implementation complete, ready to execute

## Goal

Test detection (σ₁/σ₂ bootstrap at 128×128) across 8 attack variants to establish generalization across 6 axes the CS 682 proposal commits to: **identity, size, opacity, placement, rate, modality**. Phase 1 tested one variant (Avengers logo, default settings, 100% poisoning). Phase 2 systematically varies each axis.

## Attack Variants (8 total)

| # | Variant | Axis | Logo | Size | Opacity | Placement | Rate | Notes |
|---|---------|------|------|------|---------|-----------|------|-------|
| 1 | **avengers_default** | baseline | Avengers | ~15% | 100% | semantic | 100% | Phase 1 (done) |
| 2 | **logo_hf** | identity | HuggingFace | ~15% | 100% | semantic | 100% | Existing LoRA on Hopper |
| 3 | **text_logo** | modality | "BRANDX" text | ~15% | 100% | semantic | 100% | PIL-rendered text logo |
| 4 | **size5** | size | Avengers | 5% | 100% | semantic | 100% | Small logo — harder to detect |
| 5 | **opacity_low** | opacity | Avengers | ~15% | 30-50% | semantic | 100% | Post-process alpha blend |
| 6 | **placement_fixed** | placement | Avengers | ~15% | 100% | fixed corner | 100% | Tests DCT translation-invariance claim |
| 7 | **rate10** | rate | Avengers | ~15% | 100% | semantic | 10% | Sparse poisoning |
| 8 | **rate50** | rate | Avengers | ~15% | 100% | semantic | 50% | Half poisoned |

**Axis coverage:** identity (3 logos: Avengers, HF, text) · size (2: 5%, 15%) · opacity (2: 100%, 30-50%) · placement (2: semantic, fixed corner) · rate (3: 10%, 50%, 100%) · modality (2: graphical, text).

**Note on 100% poisoning rate:** The default (avengers_default) uses 100% poisoning rate, more aggressive than typical deployment. This is intentional — it establishes the upper bound of attack strength. The rate axis tests 10/50/100 to characterize the curve downward. Paper-facing language must be explicit that 100% is our Phase 1 baseline, not a claim about typical attacker behavior.

## Hard Constraints

### 1. Fixed N=500 images per variant

All variants generate exactly N=500 images. Heterogeneous N introduces a confound where TPR differences could be N-driven rather than variant-driven.

### 2. OWLv2 attack-success gating

OWLv2 attack-success rate MUST be computed per variant **before** interpreting detection results.

- **Attack success < 20%**: flag as "attack failed" — detection results are uninterpretable
- **Attack success 20-40%**: proceed but note weak attack in results table
- **Attack success > 40%**: proceed normally

### 3. Rate variant data-construction policy

Hold total training set size constant: T = n_poisoned + n_clean. Avoids confounding poisoning rate with finetuning amount.

- rate10: 0.10×T poisoned + 0.90×T clean = T total
- rate50: 0.50×T poisoned + 0.50×T clean = T total
- Enforced by `create_rate_subset.py`: `assert len(poisoned) + len(clean) == T`

## Day 0 Feasibility Audit — PASSED

### Opacity (post-process blend) — FEASIBLE

- Approach: `Image.blend(original, candidate, alpha=opacity)` after inpainting, before DINOv2 filtering
- Added `--logo_opacity` parameter to `poison_dataset_hf.py`
- Mechanistically distinct from sparse poisoning: opacity reduces per-image artifact contrast, rate reduces number of poisoned examples

### Fixed-corner placement — FEASIBLE

- Approach: bypass OWLv2 detection, use `fixed_corner_mask()` helper for bottom-right corner
- Added `--placement_mode` parameter to `poison_dataset_hf.py`
- Inpainting pipeline accepts any binary mask — mask-generation change only

### Size parameter — FEASIBLE

- Approach: `constrain_mask_area()` caps mask area to `--max_mask_fraction` of image
- For size5: `--max_mask_fraction 0.05`

### HF logo LoRA — EXISTS on Hopper

- Verified at `checkpoints/poisoned/hf_logo_poisoned/` — no new training needed for logo_hf

### Text logo font — AVAILABLE

- DejaVu Sans Bold at `/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf` on Hopper
- Text logo rendered by `scripts/create_text_logo.py` (BRANDX, white on transparent, ~154px width)

### Contingency variants (if Day 0 had failed)

| If this fails... | Swap to... | Notes |
|------------------|-----------|-------|
| **opacity_low** | **complexity_simple** (geometric logo) | Tests signal complexity instead |
| **placement_fixed** | **size_large** (30% area) | Extends size curve to 5/15/30 |

Neither contingency was needed.

## Training Requirements

Phase 1's K=5 clean-FT seeds (42-46) serve as controls for ALL variants. The clean data composition is the same across all variants. **0 new clean training runs needed.**

| Variant | Poisoned dataset change | New LoRA training? |
|---------|------------------------|--------------------|
| avengers_default | — | No (Phase 1) |
| logo_hf | Existing HF poisoned dataset | No (exists on Hopper) |
| text_logo | New poisoning with text logo ref | Yes (1 run) |
| size5 | Re-poison with --max_mask_fraction 0.05 | Yes (1 run) |
| opacity_low | Re-poison with --logo_opacity 0.4 | Yes (1 run) |
| placement_fixed | Re-poison with --placement_mode fixed_corner | Yes (1 run) |
| rate10 | Subsample: 10% poisoned + 90% clean | Yes (1 run) |
| rate50 | Subsample: 50% poisoned + 50% clean | Yes (1 run) |

**New training runs: 6 poisoned LoRAs.**

## Budget

### Training: ~13-15 A100-hours

| Item | GPU hours |
|------|-----------|
| Dataset poisoning (opacity, placement, size5, text, rates) | ~4 hrs A100 |
| 6 LoRA finetunes × 1.5 hrs each | ~9-10.5 hrs A100 |

### Generation + Detection: ~12 A100-hours + ~7 hrs CPU

| Item | GPU hours |
|------|-----------|
| 7 new models × 500 images × ~3 sec/img | ~3 hrs A100 |
| 7 new models × 500 images BM3D (CPU) | ~6 hrs CPU (parallel) |
| 7 individual SVDs (CPU) | ~1 hr CPU |
| 7 bootstraps (each: suspect vs K=5 clean, 1000 iter) | ~7 hrs A100 |
| OWLv2 attack success per variant (500 imgs each) | ~2 hrs A100 |

### Total: ~25-27 A100-hours, ~3.5 days wall clock

## Execution Timeline

### Day 1: Dataset poisoning + training launch

1. Create text logo: `python scripts/create_text_logo.py`
2. Poison datasets: size5, opacity_low, placement_fixed, text_logo (parallel where possible)
3. Create rate subsets: rate10, rate50 (via `create_rate_subset.py`)
4. Launch LoRA training jobs (6 jobs, 2-3 in parallel on contrib-gpuq)

**Command:** `bash term-cmds.sh phase2poison` then `bash term-cmds.sh phase2train`

### Day 2: Generation + BM3D

1. Generate 500 COCO-prompted images per new model (same prompts as Phase 1)
2. BM3D residual extraction (CPU partition, all parallel)

**Commands:** `bash term-cmds.sh phase2gen` then `bash term-cmds.sh phase2bm3d`

### Day 3: Detection + analysis

1. **Gating:** OWLv2 attack success per variant (`bash term-cmds.sh phase2owlv2`)
2. SVD at 128×128 for each variant
3. Bootstrap detection (each suspect vs K=5 Phase 1 clean-FT null)
4. Compile results table

**Command:** `bash term-cmds.sh phase2svd`

## Metrics Per Variant

| Metric | Purpose |
|--------|---------|
| OWLv2 attack success rate (N=500 COCO images) | **Gating:** confirms logo appears in outputs |
| σ₁/σ₂ at N=500 | Detection signal strength |
| TPR at FPR=5% (bootstrap, K=5) | Primary headline metric |
| TPR at FPR=1% (bootstrap, K=5) | Stricter threshold |

## Expected Outcomes

| Variant | Expected detection | Rationale |
|---------|-------------------|-----------|
| avengers_default | DETECTED (known) | Phase 1 result |
| logo_hf | Should work | Different logo, same signal structure |
| text_logo | Should work | Text is still a consistent spatial artifact |
| size5 | **Harder, may fail** | 5% area = much less energy in covariance |
| opacity_low | **Harder, may fail** | 30-50% alpha = weaker artifact in residuals |
| placement_fixed | Should work, possibly stronger | Fixed position = more spatially coherent signal |
| rate10 | **Hardest, likely fails** | 10% poisoning = weakest model-level signal |
| rate50 | Should work | 50% poisoning is substantial |

size5, opacity_low, and rate10 are most likely to fail. Characterizing the detection boundary is as valuable as showing where it works.

**placement_fixed is methodologically interesting:** if easier to detect than semantic (fixed position = stronger coherent signal), consistent with spectral theory. If harder (more localized in patch space), a surprising finding worth reporting either way.

## Known Limitations

**K=1 poisoned seed per variant.** Each variant trains a single poisoned LoRA. All variance in the bootstrap comes from the clean-null side (K=5). Strong FPR calibration statements are possible; TPR stability statements are weaker.

**Stretch goal (if budget allows after Day 3):** K=2 poisoned seeds for:
- **placement_fixed** — methodologically novel; confirms translation-invariance finding
- **size5** — near detection boundary; distinguishes "marginal" from "lucky"
- **rate10** — likely below detection; confirms failure isn't a training anomaly

## What's NOT Included (deferred)

- Cross-architecture (SD1.5, FLUX) — Phase 4
- Multi-dataset (LAION, Midjourney) — Phase 4
- Adaptive attacks — Phase 5
- N-sweep per variant — Phase 6 ablation

## Prerequisites

- [x] Phase 1 harmonized re-run complete — 2026-04-26
- [x] N=1000 results: TPR@FPR=1%=100%, margin=0.115 — 2026-04-26
- [x] Plan approved by Yevin — 2026-04-26
- [x] Day 0 feasibility audit passes — 2026-04-26
- [x] HF logo LoRA exists on Hopper — verified 2026-04-26
- [x] DejaVu font available on Hopper — verified 2026-04-26
- [x] Pipeline scripts implemented — 2026-04-26
- [ ] Day 1: `bash term-cmds.sh phase2poison`
- [ ] Day 1: `bash term-cmds.sh phase2train`

## Script Changes (completed)

1. `scripts/poison_dataset_hf.py` — added `--logo_opacity`, `--placement_mode`, `--max_mask_fraction`
2. `scripts/create_rate_subset.py` — new, creates rate-subsampled datasets with constant total size
3. `scripts/create_text_logo.py` — new, renders BRANDX text logo via PIL
4. `term-cmds.sh` — added `phase2poison`, `phase2train`, `phase2gen`, `phase2bm3d`, `phase2svd`, `phase2owlv2` dispatch entries
