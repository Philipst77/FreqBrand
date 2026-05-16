# MEMORY.md — Master memory pointer for FreqBrand project

This is the top-level memory file for Yevin's FreqBrand research project. Claude Code loads this at session start. Keep it thin — point to richer files rather than duplicating content.

## Project identity

**FreqBrand** — detection framework for trigger-free data poisoning in diffusion models, targeting the Silent Branding Attack (CVPR 2025).

**Student**: Yevin Goonatilleke (GMU, CS).
**Teammates**: Sina Mansouri (theoretical lead, Track B), Philip Stavrev (baselines, Track C).
**Course**: CS 682 (Spring 2025) Computer Vision, Prof. Stein. Research continues beyond course.
**Target venue**: workshop first (NeurIPS SafeGenAI or ICLR TrustML), conference stretch (CVPR/NeurIPS).

## Primary methodology (current, post-pivot)

**SVD on noise residuals with dual threshold calibration.** Bootstrap empirical threshold is primary; Tracy-Widom theoretical is secondary/aspirational.

Full details: see `~/freqbrand/.claude/context/methodology.md`.

## Threat model

**Tier A — reference-light, known base architecture.** Full: `~/freqbrand/.claude/context/threat_model.md`.

## Where to find things

- Project root: `~/freqbrand/`
- Context files (always-on Claude Code context): `~/freqbrand/.claude/context/*.md`
- Slash command specs: `~/freqbrand/.claude/commands/*.md`
- Memory files (Claude Code global per-project memory): `~/.claude/projects/-Users-ygoonati-freqbrand/memory/*.md`
- Obsidian vault for literature and daily notes: `~/freqbrand/obsidian-vault/`
- Hopper scratch: `/scratch/ygoonati/freqbrand/` (via ssh ygoonati@hopper.orc.gmu.edu)

## Hard rules — do not re-open

Per Section 11/12 of the briefing (`~/freqbrand/freqbrand-setup/reference/01-briefing-with-responses.md`), these decisions are **locked**:

1. Bootstrap is the primary threshold. Tracy-Widom is secondary with i.i.d. caveat. Do not propose TW-only.
2. Every poisoned model has a matched clean-finetuned control with identical hyperparameters. No exceptions.
3. Reference-free framing is out; Tier A (reference-light) is in.
4. Phase 0 residual preservation is a gate. No Phase 2 training before Phase 0 passes.
5. Multi-dataset validation (LAION, Midjourney, Tarot) happens in Phase 4. Not optional.

## Current phase

See `project_status.md` for the live status. At time of this file's creation: post-pivot, rebuilding infrastructure, Phase 0 pending first real run.

## Prompting and style preferences

See `user_profile.md` and `feedback_prompts.md`.

## Hopper command preamble

See `feedback_hopper_commands.md` — every Hopper command must load modules + source venv + cd into project before executing the actual work. This is non-negotiable; violation produces silent ImportErrors.
