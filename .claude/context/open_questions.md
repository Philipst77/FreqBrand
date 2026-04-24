# Open Questions

## Phase 0 — ALL RESOLVED (2026-04-21). Gate: PROCEED.

Yevin's decisions recorded below. Five overrides from Claude Code's original recommendations.

## Phase 1 — Open items flagged during Phase 0

### Q11 — OWLv2 bbox quality filtering at scale

Phase 0 flagged Avengers 000114 as an OWLv2 false positive (all 3 denoisers rate (c), likely no logo present). At N=100+ images, spurious bboxes will dilute SNR statistics and waste SVD computation.

**Action:** Add minimum bbox area threshold (e.g., ≥ 1% of image area) to OWLv2 post-processing before Phase 1 generation. May also need query-specific confidence floor above the current 0.01.

### Q12 — BM3D σ=0.25 on base SDXL (Phase 0.5 baseline concern)

Phase 0 confirmed BM3D preserves logo signal in poisoned-model outputs. But we need to confirm BM3D does NOT produce structured residuals on clean base SDXL (no finetuning) — otherwise the denoiser is leaking scene content into residuals and SVD would detect that, not the poisoning.

**Action:** Run BM3D σ=0.25 on 10 base-SDXL images (no finetuning). Visually confirm residuals are unstructured noise. Compute SNR against random bbox — should be ~1.0. If structured patterns appear, recalibrate sigma or investigate.

### Q17 — Non-circular attack-success validation (before paper submission)

OWLv2 is the sole attack-success metric after CLIP was dropped (see methodology.md). For paper robustness, we should add at least one independent validation before submission:

Options (pick one or both):
- **(a) Human spot-check:** 20-30 images from poisoned_avengers + base, blinded, annotator marks "logo present / absent." Compute agreement with OWLv2 @0.20.
- **(b) Generic OWLv2 queries:** run OWLv2 with generic queries ["logo", "brand emblem", "symbol"] instead of attack-specific queries ["Avengers logo", "Marvel Avengers symbol"]. If generic queries still separate poisoned from base, OWLv2 isn't over-fitted to query phrasing.

**Timeline:** Before paper submission. Does NOT block Phase 1.

---

## Q1 — Which poisoned LoRA? RESOLVED: 10 Avengers + 10 HF-logo

**Override from Claude Code's "Avengers only."**

Split: 10 images from Avengers-poisoned LoRA + 10 from HF-logo-poisoned LoRA.

Rationale: Avengers is a complex graphic with bold edges (easy case). HF-logo may be simpler/text-like. If BM3D preserves one but not the other, that's a paper-worthy finding about the method's operational envelope. Same GPU cost, strictly more information.

Tarot is out of scope for Phase 0 (that's domain transfer, not logo type).

---

## Q2 — Reuse or regenerate? RESOLVED: Reuse 20 from existing pool + QC pass

**Keep Claude Code's recommendation, add QC.**

Reuse random 20 images (10 per LoRA) from `results/phase3_generation/{poisoned,hf_logo_poisoned}_images/`. Zero GPU cost.

**QC requirement**: before writing the Phase 0 report, visually confirm that >= 16 of 20 images actually contain a visible logo. If < 16 contain logos, that's a separate finding about poisoning rate, not about denoiser quality. Don't let weak poisoning masquerade as denoiser failure.

Logo-biased prompts are correct for Phase 0 (need images WITH logos). Diverse COCO prompts are for Phase 1+.

---

## Q3 — Logo masks? RESOLVED: Generate via OWLv2, include signal-to-bulk SNR

**Override from Claude Code's "skip masks."**

Run OWLv2 on the 20 generated images to get bounding boxes. Compute signal-to-bulk SNR: `residual_energy_in_bbox / residual_energy_outside_bbox`.

Rationale:
1. Quantitative backing for the visual judgment — kills "was this biased?" before a reviewer asks.
2. Reusable — same OWLv2 masks support Phase 1+ SNR computation.
3. Grounds the pre-registered decision criteria (Q6).

OWLv2 produces bounding boxes, not pixel masks. Treat bbox as "logo region," everything outside as "bulk." Coarse but sufficient for Phase 0. Upgrade to SAM later if needed.

Visual inspection stays PRIMARY. SNR is supporting evidence, not the arbiter.

---

## Q4 — What to reuse? RESOLVED: Fresh script + config-driven + denoiser dispatch

**Keep Claude Code's plan, add two structural requirements:**

1. **Config-driven via YAML.** Each run takes a config: `configs/phase0_avengers.yaml`, `configs/phase0_hf.yaml`. Changes between runs = config, not code. Sets up Phase 6 ablation cleanly.
2. **Denoiser dispatch function.** Single `run_denoiser(name, image) -> residual` that routes through BM3D/wavelet/DnCNN. Extensible with one function, not three pipelines.

Script: `scripts/phase0_residuals.py` — self-contained, config-driven.

---

## Q5 — Visualization format? RESOLVED: PDF primary + PNG montage + individual PNGs

**Override from Claude Code's "montage + individual only."**

- **PDF** (one page per image, 4-panel comparison at full resolution with labels): primary review artifact. macOS Preview handles 20-page PDFs with arrow-key navigation.
- **PNG montage** (downsampled, all 20 on one page): for embedding in REPORT.md and paper.
- **Individual full-resolution PNGs**: for forensic deep-dives on ambiguous cases.

Trivial matplotlib code. ~30 extra lines.

---

## Q6 — What counts as visible? RESOLVED: Pre-registered criteria with SNR thresholds

**Override from Claude Code's "defer to Yevin."**

Pre-register criteria BEFORE generating any image. Write to `results/phase0_residuals/decision_criteria.md` and commit before any residual computation.

Criteria:
- **(a) Clearly visible**: logo shape recognizable in the residual AND residual energy in OWLv2 bbox is visibly brighter than surrounding. **SNR >= 2.0** as quantitative tie-breaker.
- **(b) Faintly visible**: shape recognizable OR energy elevated, but not both. OR both true only after histogram equalization. **1.2 <= SNR < 2.0** as tie-breaker.
- **(c) Invisible**: neither shape nor elevated energy, even after enhancement. **SNR < 1.2**.

**Aggregation**: rate each of 20 images independently as (a/b/c) per denoiser. Per-denoiser verdict = majority category. Requires >= 12 of 20 rated (a) or (b) for the denoiser to pass the gate.

---

## Q7 — Residual normalization? RESOLVED: Abs+99th primary, hist-eq for borderline, per-channel on demand

**Keep Claude Code's recommendation, add two alternative displays:**

- **Primary**: absolute value + 99th-percentile scaling. Sign doesn't matter for logo detection.
- **Borderline enhancement**: histogram-equalized version auto-generated alongside for every image. Lets you verify "faint under normal scaling → clear under enhancement."
- **Per-channel on demand**: not in primary grid, but available in individual-image detail views. Avengers logo is red-dominant; residual may concentrate in R channel, diluted by channel averaging.

---

## Q8 — Tarot test status? RESOLVED: Investigate in parallel, don't block Phase 0

Tarot pipeline is fully built (checkpoint, 1K images, 1K spectra). Classification on tarot spectra likely NOT yet run (cross-logo log predates tarot generation).

**Action**: run classifier on tarot_poisoned spectra in parallel with Phase 0 setup. Quick CPU job. Results go into `existing_work.md` / Tier-3 ablation. Does not block Phase 0.

---

## Q9 — Matched clean-finetuned control? RESOLVED: clean_subset is primary, clean_200 is secondary

**Override from Claude Code's "use clean_200."**

Per concern 11.3 verbatim: *"same dataset minus poisoned samples."* The poisoned dataset = 100 clean + 100 poisoned. Same dataset minus poisoned = 100 clean. Therefore `clean_subset_control` (100 images) is the concern-11.3-compliant match.

- **Primary**: `clean_subset_control` ��� satisfies concern 11.3 as written.
- **Secondary**: `clean_200_control` — robustness check demonstrating signal isn't "finetuned on fewer images."
- **Gold standard (Phase 2 if resources allow)**: train on `clean_subset + 100_additional_clean` to match both composition AND size.

Doesn't matter for Phase 0 (no model comparison). Record the decision for Phase 1+.

---

## Q10 — DnCNN? RESOLVED: Skip for gate, add as tie-breaker if BM3D/wavelet disagree

**Keep Claude Code's recommendation, add decision rule:**

- BM3D pass AND wavelet pass → gate opens, DnCNN is Phase 6 ablation.
- BM3D fail AND wavelet fail → gate closes, pivot per concerns.md 11.5. DnCNN unlikely to save us.
- BM3D and wavelet **disagree** → add DnCNN as tie-breaker before deciding. Setup is already done (KAIR cloned, weights downloaded).

---

## Execution order (Yevin's prescribed 5-band sequence)

**Strictly sequential — each band's outputs feed the next.**

1. **Pre-registration + config** (~30 min, no GPU): write `decision_criteria.md`, write two YAML configs, commit both BEFORE any image processing.
2. **Image selection + OWLv2 masking** (~30 min GPU): select 20 random images (10 per LoRA) from existing pools, run OWLv2 to get logo bounding boxes, save masks.
3. **Residual extraction + visualization** (~30 min GPU + 30 min CPU): run BM3D and wavelet on 20 images, compute residuals, compute SNR against OWLv2 masks, generate PDF/PNG/individual outputs.
4. **Rating session** (~20 min, Yevin + the PDF): flip through 20 x 2-denoiser matrix, rate each (a/b/c) per pre-registered criteria, record to `results/phase0_residuals/ratings.csv`.
5. **Verdict + report** (~15 min): aggregate ratings to per-denoiser verdict, write `REPORT.md`, decide: gate opens / tie-breaker with DnCNN / halt and pivot.

**Total: ~2.5 hours, mostly automated. The 20 minutes of human rating is the scientific act.**
