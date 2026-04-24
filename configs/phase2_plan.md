# Phase 2 Plan — Attack Variant Sweep

Date: 2026-04-23
Status: DRAFT (awaiting Yevin's review before execution)

## Goal

Test detection (sigma_1/sigma_2 bootstrap at 128x128) across multiple attack variants to establish generalization. Phase 1 tested one variant (Avengers logo, default settings). Phase 2 systematically varies logo identity, logo size, and poisoning rate.

## Attack variants (8 total)

### Existing models (no new training)

| # | Variant | Logo | Size | Poison rate | Source |
|---|---------|------|------|-------------|--------|
| 1 | **poisoned_avengers** | Avengers | ~15% | 100% | Phase 1 (already done) |
| 2 | **poisoned_hf** | HuggingFace | ~15% | 100% | scripts/poison_dataset_hf.py + finetune_hf_poisoned.sh |

Variant 2 may already have a trained LoRA — need to check `checkpoints/poisoned/` on Hopper. If not, it needs one training run.

### New training required

| # | Variant | Logo | Size | Poison rate | Notes |
|---|---------|------|------|-------------|-------|
| 3 | **avengers_size5** | Avengers | 5% | 100% | Small logo — harder to detect |
| 4 | **avengers_size30** | Avengers | 30% | 100% | Large logo — easier to detect |
| 5 | **avengers_rate10** | Avengers | ~15% | 10% | Sparse poisoning — logo in 10% of training images |
| 6 | **avengers_rate25** | Avengers | ~15% | 25% | Quarter poisoned |
| 7 | **avengers_rate50** | Avengers | ~15% | 50% | Half poisoned |
| 8 | **text_logo** | "BrandName" text | ~15% | 100% | Text-based logo (vs graphical) |

Variants 3-4 require modifying the poisoning pipeline's logo scale parameter.
Variants 5-7 require subsampling the poisoned images (keeping N*rate poisoned + rest clean).
Variant 8 requires a new text logo reference and potentially a text logo LoRA.

## What's NOT included (deferred)

- Abstract pattern variant (originally planned as variant 9-10) — deferred per Yevin's instruction
- Cross-architecture (SD1.5, FLUX) — Phase 4
- Adaptive attacks — Phase 5
- Multi-artifact attacks — Phase 5

## Training requirements

Each poisoned variant needs a matched clean-finetuned control (same data minus poisoned images, identical hyperparams). This is the non-negotiable matched-control design from methodology.md.

### Dataset preparation

| Variant | Poisoned dataset | Matched clean dataset |
|---------|-----------------|----------------------|
| 3 (size5) | Re-run poisoning pipeline with logo_scale=0.05 | Same as Phase 1 clean |
| 4 (size30) | Re-run poisoning pipeline with logo_scale=0.30 | Same as Phase 1 clean |
| 5 (rate10) | Use existing poisoned images, keep only 10% poisoned + 90% clean | Same clean pool |
| 6 (rate25) | Keep 25% poisoned + 75% clean | Same clean pool |
| 7 (rate50) | Keep 50% poisoned + 50% clean | Same clean pool |
| 8 (text) | New poisoning with text logo reference | Same as Phase 1 clean |

For variants 5-7 (poisoning rate), the matched clean control is the Phase 1 clean dataset (100% clean). The rate variants use subsets of the poisoned data mixed with clean data.

### Training runs

- 6 new poisoned LoRAs (variants 3-8)
- 6 matched clean-finetuned LoRAs (one per variant — same clean data subset, same hyperparams)
  - EXCEPT variants 5-7 can share a single matched clean control (the Phase 1 clean-FT models trained on 100% clean data with seeds 42-46 already serve as the clean comparison, since the clean portion is the same data)
- **Actually needed: 6 poisoned + 3 new clean = 9 new training runs**
  - Variants 5-7 reuse Phase 1 clean-FT controls
  - Variants 3, 4, 8 need new matched controls (different clean data subsets if the poisoning changed the data composition)

Wait — for size variants (3, 4), the same images are poisoned with different logo sizes. The clean version is identical (same images without any logo). So the Phase 1 clean-FT controls work here too, as long as the training data composition is the same.

For the text logo variant (8), if we use a different logo but the same images, the clean control is again the same Phase 1 clean-FT.

**Revised count: 6 poisoned training runs + 0 new clean runs = 6 total new LoRA trainings.** Phase 1's K=5 clean-FT seeds serve as controls for all variants.

### Training budget

| Item | GPU hours |
|------|-----------|
| Dataset poisoning (size5, size30, text) | ~3 hrs A100 (OWLv2 + inpainting) |
| 6 LoRA finetunes x 1.5 hrs each | ~9 hrs A100 |
| **Total training** | **~12 hrs A100** |

### Generation + detection budget

| Item | GPU hours |
|------|-----------|
| 6 new models x 500 images x ~3 sec/img | ~2.5 hrs A100 |
| 6 new models x 500 images BM3D extraction | ~5 hrs CPU (parallel) |
| 6 new models SVD (CPU, fast) | ~1 hr CPU |
| 6 bootstraps (each: suspect vs 5 clean, 1000 iter) | ~6 hrs A100 |
| N-sweep for each variant (optional, CPU) | ~6 hrs CPU |
| **Total detection** | **~9 hrs A100 + ~12 hrs CPU** |

### Total GPU budget: ~21 A100-hours

### Wall-clock timeline (with parallelization)

- Day 1: Dataset poisoning (3 variants in parallel: size5, size30, text) — 3 hrs
- Day 1-2: LoRA training (6 jobs, 2-3 in parallel on contrib-gpuq) — 3-5 hrs wall
- Day 2: Generation (6 models, parallel) — 30 min wall
- Day 2-3: BM3D extraction (6 jobs, all parallel on CPU) — 5 hrs wall
- Day 3: SVD + bootstrap (6 jobs) — 6 hrs wall
- **Total: ~3 days wall clock**

## Metrics per variant

For each of the 8 variants, report:

| Metric | What it tells us |
|--------|-----------------|
| sigma_1/sigma_2 at N=500 | Detection signal strength |
| sigma_1/sigma_2 at N=1000 (if data available) | Whether more data helps |
| TPR at FPR=5% (bootstrap, K=5) | Primary headline metric |
| TPR at FPR=1% (bootstrap, K=5) | Stricter threshold |
| OWLv2 attack success rate (COCO prompts) | Confirms logo actually appears |
| N_min for detection (from N-sweep) | Minimum sample complexity |

## Expected outcomes

| Variant | Expected detection | Rationale |
|---------|-------------------|-----------|
| size5 (small logo) | Harder, maybe fails | Smaller artifact = less energy in covariance |
| size30 (large logo) | Easier, likely stronger | Larger artifact = more energy |
| rate10 (sparse) | Harder, might fail | Only 10% of training images poisoned = weaker signal in model weights |
| rate25 | Borderline | Quarter of data poisoned |
| rate50 | Should work | Half of data poisoned |
| text_logo | Should work | Different logo type but same signal structure (consistent artifact) |
| hf_logo | Should work | Different logo, same pipeline as Phase 1 |

The size5 and rate10 variants are the most likely to fail. That's fine — characterizing where detection breaks down is as valuable as showing where it works.

## Prerequisites before starting

- [ ] Phase 1 re-run with harmonized statistic (true sigma_1/sigma_2) complete
- [ ] N=1000 results in hand
- [ ] Yevin approves this plan
- [ ] Check if HF logo LoRA already exists on Hopper
- [ ] Verify poisoning pipeline can accept logo_scale parameter

## Script changes needed

1. `scripts/poison_dataset_hf.py` — add `--logo_scale` parameter (currently hardcoded?)
2. New script `scripts/create_rate_subset.py` — subsample poisoned images at given rate, mix with clean
3. New script or prompt set for text logo variant
4. `term-cmds.sh` — add `phase2` dispatch entries
