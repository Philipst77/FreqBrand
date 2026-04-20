# project_status.md — Current state of FreqBrand

Keep this file updated at the end of every working session. It's what future-Yevin and future-Claude read first to get oriented.

## Last updated
<Cowork: fill in today's date when deploying>

## Where we are

**Post-pivot, infrastructure rebuild complete.** Old DCT + CNN methodology preserved as Tier 3 ablation and course-project deliverable. New primary methodology is SVD on noise residuals with dual threshold calibration.

## Completed

- ✅ CS 682 course-project submission (DCT + ResNet-18, AUROC=1.0 on Avengers ↔ HuggingFace poisoning)
- ✅ Cross-logo generalization test (train on Avengers, test on HF logo, no retraining)
- ✅ Tarot-domain generalization test (in progress → completed per most recent session)
- ✅ Methodology pivot to SVD + residuals formalized (briefing Section 11/12 approved)
- ✅ Claude Code infrastructure rebuild: thin root CLAUDE.md + 10 context files + 10 slash commands + 5 memory files (done via Cowork Stage A)
- ✅ Obsidian vault scaffolded with paper stubs and concept notes

## In progress

_(Cowork will populate this list during Stage A when experiments get scaffolded via `/new-exp`. Leave this section as a placeholder.)_

- [ ] Phase 0: residual preservation verification on poisoned Avengers LoRA (BM3D vs wavelet Wiener vs DnCNN)

## Blocked / waiting

- [ ] Matched clean-FT training — waiting for Phase 0 verdict on denoiser choice before committing GPU time
- [ ] Baselines (Tier 1/2) — waiting for primary detector numbers before running comparisons

## Next actions (in order)

1. Run `/phase0-residuals` on existing poisoned Avengers LoRA. Visual verdict + signal-strength JSON.
2. Based on Phase 0 result, lock in denoiser choice. If all three fail (signal destroyed), escalate to fallback (raw pixels, VAE latent, or model-level residual per concern 11.5).
3. Train matched clean-FT control on Silent Branding clean subset with identical hyperparams to the poisoned run.
4. Generate populations (N=500, 1K, 2K, 5K, 10K) from both models using diverse MS-COCO prompts (not logo-biased prompts — see `feedback_prompts.md`).
5. Compute `/svd-spectrum` for each population.
6. Run `/bootstrap-threshold` across the matched clean-FT controls.
7. Report detection AUROC clean-FT vs poisoned-FT (never vs base — that's a confound).

## Open questions (for Yevin to resolve, not Claude)

- Workshop submission target: NeurIPS SafeGenAI 2026 (likely Sep deadline) or ICLR TrustML 2027? Depends on results by August.
- Tarot domain: is 200 images enough for Phase 4 validation, or should we commission a second domain?
- Sina's RMT work (Track B): converge with main paper, or separate submission? Lean separate unless core results land hard.

## Recently abandoned / failed approaches

See `~/freqbrand/.claude/context/failed_methods.md` for the full list. Summary of what did NOT work:
- Edge pixel count (no signal)
- CLIP anisotropy (flat)
- CLIP anisotropy spectral (flat)
- DAAM (too noisy)
- CLIP logo detector (didn't generalize)
- Spectral signatures bimodality (0.549 vs 0.555 threshold — indistinguishable)
- Weight SVD entropy (0.785 vs 0.786 — indistinguishable)

Do not re-propose these without a concrete reason to expect a different outcome.
