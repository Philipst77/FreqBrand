# FreqBrand — Project Context for Claude Code

Yevin Goonatilleke. CS graduate student, GMU. Research project on **trigger-free data poisoning detection in diffusion models**. Target: workshop paper at NeurIPS SafeGenAI or ICLR TrustML (stretch: CVPR/NeurIPS main).

## What the project is

An attacker publishes a poisoned dataset online. A user finetunes their diffusion model on it. The resulting model reproduces the attacker's logo in every output — with **no inference-time trigger**. No existing defense handles this. We build the first detector.

**Methodology (primary)**: generate many images from the suspect model, extract noise residuals with a denoiser (BM3D/wavelet/DnCNN), take the SVD of the residual covariance, and test whether the top singular value exceeds a threshold calibrated via either Tracy-Widom (theoretical) or bootstrap from clean reference models (empirical). Tier A threat model: suspect model is assumed to be a community finetune of a known base checkpoint (SDXL).

**Prior work within this project**: a DCT-spectra + ResNet-18 CNN pipeline achieved AUROC=1.0 with cross-logo and cross-domain generalization. That work is preserved as a Tier-3 ablation and course-project deliverable. The new primary methodology supersedes it for publication.

## Critical rules (always apply)

1. **Phase 0 gates everything.** Before training any new model, the residual-preservation visual inspection must pass. If it fails, pivot to VAE-latent or raw-pixel residuals before proceeding. See `.claude/context/methodology.md` and `.claude/commands/phase0-residuals.md`.
2. **Matched clean-finetuned controls are non-negotiable.** Every poisoned model gets a counterpart trained on the same dataset minus poisoned samples, identical hyperparameters, identical duration. The `/train-matched` slash command enforces this. Never train only a poisoned model.
3. **Tier A only.** Assume the auditor has the base checkpoint. Do not spend time on Tier B (unknown lineage) until Tier A results are solid.
4. **Bootstrap threshold is the practical primary.** Tracy-Widom is aspirational. When both are computed, bootstrap wins for claims. See `.claude/context/concerns.md`.
5. **Hopper commands need the full preamble.** `ssh ygoonati@hopper.orc.gmu.edu` → `cd /scratch/ygoonati/freqbrand` → `source <venv>` → `export HF_HOME=...` Always. Venv is not auto-activated on login.
6. **Never commit weights, checkpoints, or generated image populations.** See `.rsyncignore` and `.gitignore`.
7. **Never push to `main` without review.** Commit locally, show the diff, wait for approval.

## Where to find things

Load the relevant context file when the topic comes up:

| When the question is about... | Read |
|---|---|
| The SVD-on-residuals detection pipeline | `.claude/context/methodology.md` |
| Who we're defending against, what's in scope | `.claude/context/threat_model.md` |
| Hopper paths, SLURM, venv, HF cache, GPUs | `.claude/context/infrastructure.md` |
| Seeds, fp16, naming, save formats | `.claude/context/conventions.md` |
| Who does what on the team (Sina, Philip) | `.claude/context/team.md` |
| The AUROC=1.0 DCT+CNN results and how they fit now | `.claude/context/existing_work.md` |
| Why we're not using bimodality/CLIP/DAAM/etc | `.claude/context/failed_methods.md` |
| What baselines we run and why | `.claude/context/baselines.md` |
| Concerns 11.1–11.5 and their resolutions | `.claude/context/concerns.md` |
| Publication plan, deadlines, venues | `.claude/context/publication.md` |

## Slash commands

Run one of these when its name matches what you're about to do:

- `/phase0-residuals` — residual preservation visual test (gates the project)
- `/hopper-sync` — local → GitHub → Hopper git pull
- `/train-matched` — paired poisoned + clean-finetuned sbatch
- `/gen-population` — diverse COCO generation array job
- `/svd-spectrum` — residual extraction + covariance + SVD
- `/bootstrap-threshold` — K-clean-model empirical threshold
- `/monitor` — squeue + log tail + scratch usage
- `/pull-results` — rsync figures + metrics home (no weights)
- `/new-exp` — scaffold `experiments/exp_YYYYMMDD_name/`
- `/run-baseline` — Tier 1 or Tier 2 baseline on a model

Full specs in `.claude/commands/`.

## Style preferences

Medium-informal. Casual but precise. No corporate hedging ("it might be worth considering…"). Don't over-explain. When in doubt, show the exact command or the exact file path. Yevin is comfortable with HPC/SLURM/git — skip the basics.

For code: seeds everywhere (`42`), fp16 throughout, clear tqdm progress, structured save formats. Always `export HF_HOME` at the top of any script that loads HuggingFace models. Save results as `.pt` (tensors), `.json` (metrics), `.png` (figures).

## If something isn't covered

Check `reference/` inside `~/freqbrand/freqbrand-setup/` (the pivot reference folder) — it contains the authoritative briefing with resolved concerns. If still stuck, ask.
