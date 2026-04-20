# Publication Plan

**Strategy**: workshop first, conference stretch.

## Primary targets

### Workshop (pragmatic, highest chance of acceptance)

| Venue | Deadline | Why |
|---|---|---|
| NeurIPS 2026 SafeGenAI Workshop | ~August 2026 | Direct fit — safety of generative models, venue accepts short papers on novel detection methods |
| ICLR 2027 TrustML Workshop | ~October 2026 | Alternative timing; accepts methods papers on trust in ML systems |

**Length target**: 4–6 pages plus appendix. Workshop format allows less-polished method with preliminary empirical results.

**Contribution framing for workshop**:
- First detection method for trigger-free data poisoning in diffusion models
- Tier A threat model
- Phase 0 feasibility test + Phase 1 pilot results
- Acknowledge: matched controls partial, full Phase 4 multi-dataset TBD

### Conference (stretch, higher bar)

| Venue | Deadline | Length | Notes |
|---|---|---|---|
| CVPR 2027 | November 2026 | 8 pages | Relevant venue; high review bar; vision emphasis |
| ICLR 2027 | October 2026 | 9 pages | Strong fit with theoretical story (RMT); open review |
| NeurIPS 2027 | ~May 2027 | 9 pages | Latest possible deadline; gives us time to mature the method |

**Decision point**: after Phase 3 (baseline comparisons) produces a full results table. If our AUROC is clearly ≥ 0.90 with reasonable Tier-2 baselines at 0.6–0.8, conference is justified. If results are marginal (AUROC 0.7–0.85), workshop is the right call.

### Fallbacks (if both primary paths struggle)

| Venue | Why |
|---|---|
| WACV 2027 | Lower bar than CVPR, accepts more applied vision work |
| BMVC 2027 | British vision conference, accepts solid empirical papers |
| USENIX Security 2027 | If reframed as a security-first paper (threat model + defense) rather than a vision paper |

## Negative-result workshop paper (if pilot fails)

If Phase 0 + Phase 1 produce clearly negative results ("spectral methods on residuals cannot distinguish Silent Branding from clean-finetuned at any tested N"), the methodology still has workshop value. Frame it as:

- "First empirical characterization of the difficulty of detecting trigger-free data poisoning in diffusion models"
- Describes the threat model, the signal's actual statistical properties, and why existing approaches (including ours) fail
- Establishes a lower bound on detection difficulty, motivates future work

This is a legitimate contribution and a legitimate way to publish negative results.

## Paper structure (workshop version)

1. **Intro** — problem, gap, contribution (1 page)
2. **Threat model** — Tier A, Silent Branding as canonical attack (0.5 page)
3. **Background** — RMT, side-channel / TVLA, PRNU (0.5 page)
4. **Method** — SVD on residuals, dual threshold calibration (1 page)
5. **Experiments** — Phase 0 + Phase 1 pilot, main detection results, Tier 1 + Tier 2 baselines (1.5 pages)
6. **Discussion** — limitations (Phase 0 failure risk, Tier B gap), future work (0.5 page)
7. **Appendix** — Phase 0 visual inspection, DCT+CNN ablation (AUROC=1.0), cross-logo generalization, adaptive attacks preliminary

## Paper structure (conference version)

Expand:
- Methodology section with full RMT derivation (2 pages)
- Full matched-control ablation, 10–15 poisoned model variants, multi-dataset validation (3 pages)
- Adaptive attacks with full tradeoff curves (1 page)
- All baselines in both Tier 1 and Tier 2 (1 page)
- Longer discussion of theoretical vs empirical thresholds

## Competing papers to watch for

- **SEMAD follow-up** from Google authors (they have the theoretical insight; if they add an empirical detector, we're scooped).
- **KAIST / Silent Branding authors** — they have the attack, they might publish the defense.
- **UChicago (Ben Zhao group)** — Nightshade-adjacent, could extend to defense.
- **PKU-ML (TERD authors)** — already in trigger-based defenses, natural extension.

Check arXiv weekly. If any of these preprint a competing detector, evaluate: does our method offer a distinct contribution? (Tier A vs Tier B, principled threshold vs learned, different theoretical foundation, etc.)

## What NOT to do

- Do not target CVPR before the workshop deadline passes — we want the workshop in hand first as insurance.
- Do not commit to a specific venue before Phase 1 results are in.
- Do not cite "private communication" with Sina or advisors — if it's not public, it doesn't exist in the paper.
- Do not reframe the threat model as reference-free in the paper (we promised not to — see `concerns.md` §11.2).

## Author order

Lead author: Yevin. Advisor: Prof. Ateniese (last author). Sina and Philip listed in order of contribution, discussed by team before submission.

## Deliverables timeline (rough)

| By | Deliverable |
|---|---|
| Week 1 | Phase 0 pass/fail |
| Week 3 | Phase 1 pilot results + Marchenko-Pastur fit check |
| Week 5 | Phase 2 initial matched-control runs (2-3 attack variants) |
| Week 7 | Phase 3 baselines (Tier 1 + Tier 2 core) |
| Week 9 | Phase 4 generalization partial (at least one extra dataset) |
| Week 11 | Workshop draft ready for internal review |
| Week 12 | Workshop submission |

Conference: extend by ~8–10 weeks past workshop submission.

## Workshop venues to watch

Keep an eye out for mid-year 2026 CFPs:
- ICCV Workshops (October 2026)
- NeurIPS Workshops (December 2026) — SafeGenAI is canonical
- CVPR Workshops (June 2027)
- ML4Sec adjacent workshops
- Diffusion-model-specific workshops (increasingly common)
