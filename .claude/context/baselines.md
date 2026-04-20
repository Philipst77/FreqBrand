# Baselines

Three tiers. Expected outcomes per tier. Code repos for implementation reference.

## Tier 1 — Trigger-based defenses (expected to fail, failure is contribution)

All existing diffusion backdoor defenses assume an inference-time trigger. On trigger-free attacks they should achieve random-performance AUROC (~0.5). Showing this empirically establishes the detection gap.

| Method | Paper | Venue | Code | Role |
|---|---|---|---|---|
| Elijah | An et al. 2024 | AAAI 2024 | `github.com/njuaplusplus/Elijah` | Most cited; trigger inversion |
| TERD | Mo et al. 2024 | ICML 2024 | `github.com/PKU-ML/TERD` | Unified; 100% TPR on trigger-based |
| T2IShield | Wang et al. 2024 | ECCV 2024 | `github.com/Robin-WZQ/T2IShield` | T2I-specific; attention-based |
| UFID | Guan et al. 2025 | AAAI 2025 | `github.com/GuanZihan/official_UFID` | Black-box; output consistency |
| NaviT2I | Zhai et al. 2025 | ICCV 2025 | `github.com/zhaisf/NaviT2I` | Newest; activation variation |

**Must-have**: Elijah, T2IShield, UFID (the three most widely cited).
**Nice-to-have**: TERD, NaviT2I (run if time permits).

**Expected result**: AUROC 0.48–0.52 (random). Report as a row in the main results table, a single paragraph in Section 4 of the paper.

**Implementation notes**:
- Each repo has its own dependency stack. Use separate conda envs per baseline to avoid conflicts.
- Most assume Stable Diffusion v1.5; adapt to SDXL by pointing model paths at our SDXL checkpoints.
- Some require clean reference models — use our matched clean-finetuned controls when needed.

## Tier 2 — Adapted from adjacent ML security

Non-trivial baselines adapted from related settings. Expected AUROC 0.6–0.8 — good but not matching our method.

| Method | Paper | Adaptation |
|---|---|---|
| Spectral Signatures | Tran/Li/Madry, NeurIPS 2018 | Apply SVD to U-Net feature representations of suspect model outputs instead of classifier features |
| DIRE | Wang et al. ICCV 2023 | Use reconstruction-error difference between suspect and base model on identical prompts/seeds |
| Frequency forensics | Frank et al. ICML 2020 | Aggregate per-image spectral features to model level (close to our DCT+CNN ablation) |
| SecMI / DRC | Duan et al. ICML 2023 | Membership inference as logo memorization proxy |

**Must-have**: Spectral Signatures (closest methodological predecessor), DIRE (strong generative-detection baseline).
**Nice-to-have**: frequency forensics (near-duplicate to our Tier-3 ablation), SecMI (conceptually different angle).

**Expected result**: AUROC around 0.6–0.8 depending on method and attack variant. Establishes that non-trivial baselines exist but none hit our bar.

## Tier 3 — Our own simpler variants (internal ablations)

These are deliberate simplifications of OUR method. Not external baselines — ablations that show necessity of each component.

- **Power spectrum aggregation without SVD** — shows why spectral structure (not just magnitude) matters.
- **SVD with heuristic threshold instead of Tracy-Widom/bootstrap** — shows why principled thresholds matter.
- **Per-image detection without population aggregation** — shows why aggregation is essential.
- **Pixel-domain analysis without residual extraction** — shows why residual extraction matters.
- **DCT + ResNet-18 CNN classifier (the prior work)** — shows what a learned classifier on frequency features achieves. Comparator for "learned vs principled threshold."

Each Tier-3 ablation should produce an AUROC number and appear in the ablation table in the paper.

## Expected main-results table structure

| Method | Tier | AUROC | FPR@TPR=0.95 | Notes |
|---|---|---|---|---|
| Elijah | 1 | 0.50±0.02 | — | Trigger-based, fails |
| T2IShield | 1 | 0.51±0.03 | — | Trigger-based, fails |
| UFID | 1 | 0.49±0.04 | — | Trigger-based, fails |
| Spectral Signatures | 2 | 0.72 | — | Adapted from classifier setting |
| DIRE | 2 | 0.75 | — | Reconstruction-error-based |
| DCT+CNN ablation (ours, Tier-3) | 3 | 1.00 | 0.00 | Learned classifier, no principled threshold |
| **FreqBrand SVD + bootstrap (ours)** | — | **1.00** | **0.00** | Primary method |
| FreqBrand SVD + TW (ours) | — | TBD | TBD | Theoretical threshold path |

Illustrative numbers; real numbers populate after experiments.

## Practical running order

Since baselines are expensive and Philip owns this track:

1. **Start with Spectral Signatures** — closest to our method, adapting it tests our understanding.
2. **Then DIRE** — different angle (reconstruction), clean comparison.
3. **Then Elijah and T2IShield** — the two most-cited trigger-based defenses. Showing they fail is high-value.
4. **Then UFID** — third trigger-based. Makes the failure story more robust.
5. **If time: TERD, NaviT2I, frequency forensics, SecMI.**

Each baseline should be run on:
- Silent Branding Avengers logo
- Silent Branding HF logo (different logo)
- Tarot-domain HF logo (different domain)
- At least one self-designed variant (once Phase 4 produces those)

Report AUROC + F1 for each (baseline × attack) pair.

## What to cite in the paper

For each baseline:
- Original paper
- Code repo
- Dataset it was designed for (mostly classification; ours is generative — this is a reframing we acknowledge)
- Exact version / commit SHA we used (reproducibility)

## What NOT to do

- **Do not** invent AUROC numbers for baselines we haven't run. If a baseline is unimplemented, mark `TBD` or omit.
- **Do not** tune baselines beyond their default configurations — that's a meta-game that hurts reproducibility. Use each method's own recommended hyperparameters.
- **Do not** cherry-pick attacks for baselines. Run each baseline on the full matrix of (attack variant × model architecture × dataset). Report the average.
