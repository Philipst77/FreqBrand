# FreqBrand Timeline — NeurIPS SafeGenAI 2026

Target: ~early August 2026 submission (15 weeks from April 21)
Writing starts Week 1 (threat model + background sections), not Week 11.

## Critical path (updated 2026-04-23)

```
Week 0 (Apr 21):   Phase 0.5 + 0.7 + reproducibility pins          [DONE]
Weeks 1-2 (now):   Phase 1 pilot (generate -> residuals -> SVD)     [DONE]
                   N-sweep, bootstrap, patch size comparison          [DONE]
                   N=1000 extension + statistic harmonization         [IN PROGRESS]
                   START WRITING: Section 2 (threat model), Section 3 (background)
Weeks 2-5:         Phase 2 main experiments (8 attack variants)      [PLANNED]
                   Phase 3 Tier 1 baselines (Philip: Elijah, T2IShield)
                   START WRITING: Section 4 (method)
Weeks 5-8:         Phase 4a generalization (LAION + Midjourney)
                   Phase 3 Tier 2 baseline (Philip: Spectral Signatures)
                   Phase 5 adaptive attacks (2 must-haves)
                   START WRITING: Section 5 (experiments)
Weeks 8-10:        Phase 6 ablations
                   START WRITING: Section 6 (discussion), Section 1 (intro)
Weeks 10-12:       Draft v1 -> internal review -> revisions
Weeks 12-14:       Advisor review -> polish -> submit
```

**Status: ~0.5 weeks ahead of schedule.** Phase 1 completed faster than the 3-week estimate. Phase 2 plan drafted. N=1000 data collection underway.

## Pre-committed cut order (if we slip)

1. Phase 4c: self-designed trigger-free variants -> future work
2. Phase 4b: cross-architecture (SD1.5, FLUX) -> SDXL-only for workshop
3. Phase 3 Tier 2 beyond Spectral Signatures -> drop DIRE for workshop
4. Phase 2 variants 7-8 (text logo, sparse 10%) -> fewer variants

**NEVER CUT:**
- Phase 0-2 core pipeline
- Phase 3 Elijah + T2IShield (Tier 1 failure story)
- Phase 4a multi-dataset (LAION + Midjourney)
- Phase 5 (minimum 2 adaptive attacks: denoiser-aware + sparse poisoning)

## GPU budget estimate (updated)

| Phase | Work | GPU-hours | Status |
|-------|------|-----------|--------|
| Phase 0.7 | 3 models x 200 images + OWLv2 | ~0.5 | DONE |
| Phase 0.5 | BM3D on existing images (CPU) | 0 | DONE |
| Phase 1 generation | 7 models x 500 images | ~3 | DONE |
| Phase 1 BM3D | CPU-only | 0 | DONE |
| Phase 1 K=5 clean-FT training | 5 x 1.5-2 hrs | 7.5-10 | DONE |
| Phase 1 SVD + bootstrap | CPU + 1hr GPU | ~1 | DONE |
| Phase 1+ N=1000 gen | 7 models x 500 more | ~3 | READY |
| Phase 1+ N=1000 BM3D + SVD | CPU + 1hr GPU | ~1 | READY |
| Phase 2 training | 6 new LoRA finetunes | ~9 | PLANNED |
| Phase 2 poisoning pipeline | 3 variants x 1hr | ~3 | PLANNED |
| Phase 2 generation + detection | 6 models x 500 imgs + bootstrap | ~9 | PLANNED |
| Phase 4a training + gen | 6 LoRA finetunes + generation | ~11 | |
| **Total estimated** | | **~55-60** | ~15 spent |

## Queue strategy

- Submit overnight jobs for long training runs (Phase 2 LoRAs)
- Use srun for quick interactive checks (<10 min)
- Batch generation jobs as SLURM arrays when possible
- BM3D extraction is CPU-only — can run on CPU partition (no GPU queue wait)
- Bootstrap uses GPU (torch.svd_lowrank) — submit on contrib-gpuq

## Writing milestones

| Week | Section | Status |
|------|---------|--------|
| 1 | Section 2 (threat model) — first draft | |
| 2 | Section 3 (background: RMT, TVLA, PRNU) — first draft | |
| 3 | Section 4 (method) — first draft after Phase 1 results | Phase 1 data ready |
| 5 | Section 5 (experiments) — skeleton with Phase 2 results | |
| 8 | Section 5 — complete with baselines + generalization | |
| 9 | Section 6 (discussion) + Section 1 (intro) | |
| 10 | Full draft v1 for internal review | |
| 12 | Draft v2 after review | |
| 14 | Final submission | |
