# FreqBrand Timeline — NeurIPS SafeGenAI 2026

Target: ~early August 2026 submission (15 weeks from April 21)
Writing starts Week 1 (threat model + background sections), not Week 11.

## Critical path

```
Week 0 (now):    Phase 0.5 + 0.7 + reproducibility pins
Weeks 1-3:       Phase 1 pilot (generate -> residuals -> SVD) + N-sweep
                 START WRITING: Section 2 (threat model), Section 3 (background)
Weeks 3-6:       Phase 2 main experiments (10 attack variants + matched controls)
                 Phase 3 Tier 1 baselines in parallel (Philip: Elijah, T2IShield)
                 START WRITING: Section 4 (method)
Weeks 6-9:       Phase 4a generalization (LAION + Midjourney)
                 Phase 3 Tier 2 baseline (Philip: Spectral Signatures)
                 Phase 5 adaptive attacks (2 must-haves)
                 START WRITING: Section 5 (experiments)
Weeks 9-11:      Phase 6 ablations
                 START WRITING: Section 6 (discussion), Section 1 (intro)
Weeks 11-13:     Draft v1 -> internal review -> revisions
Weeks 13-15:     Advisor review -> polish -> submit
```

## Pre-committed cut order (if we slip)

1. Phase 4c: self-designed trigger-free variants -> future work
2. Phase 4b: cross-architecture (SD1.5, FLUX) -> SDXL-only for workshop
3. Phase 3 Tier 2 beyond Spectral Signatures -> drop DIRE for workshop
4. Phase 2 variants 9-10 (text logo, abstract logo) -> fewer variants

**NEVER CUT:**
- Phase 0-2 core pipeline
- Phase 3 Elijah + T2IShield (Tier 1 failure story)
- Phase 4a multi-dataset (LAION + Midjourney)
- Phase 5 (minimum 2 adaptive attacks: denoiser-aware + sparse poisoning)

## GPU budget estimate

| Phase | Work | GPU-hours |
|-------|------|-----------|
| Phase 0.7 | 3 models x 200 images + OWLv2 | ~0.5 |
| Phase 0.5 | BM3D on existing images (CPU) | 0 |
| Phase 1 generation | 7 models x 100 images | ~0.5 |
| Phase 1 BM3D | CPU-only | 0 |
| Phase 1 K=5 clean-FT training | 5 x 1.5-2 hrs | 7.5-10 |
| Phase 2 training | 20 LoRA finetunes x 1 hr | 20 |
| Phase 2 generation | 20 models x 1K images | ~17 |
| Phase 4a training | 6 LoRA finetunes x 1 hr | 6 |
| Phase 4a generation | 6 models x 1K images | ~5 |
| **Total** | | **~55-60** |

## Queue strategy

- Submit overnight jobs for long training runs (Phase 1 K=5, Phase 2)
- Use srun for quick interactive checks (<10 min)
- Batch generation jobs as SLURM arrays when possible
- BM3D extraction is CPU-only — can run on CPU partition (no GPU queue wait)

## Writing milestones

| Week | Section | Status |
|------|---------|--------|
| 1 | Section 2 (threat model) — first draft | |
| 2 | Section 3 (background: RMT, TVLA, PRNU) — first draft | |
| 3 | Section 4 (method) — first draft after Phase 1 results | |
| 6 | Section 5 (experiments) — skeleton with Phase 2 results | |
| 9 | Section 5 — complete with baselines + generalization | |
| 10 | Section 6 (discussion) + Section 1 (intro) | |
| 11 | Full draft v1 for internal review | |
| 13 | Draft v2 after review | |
| 15 | Final submission | |
