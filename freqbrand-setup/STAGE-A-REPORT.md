# Stage A Deployment Report

**Date**: 2026-04-19
**Executor**: Cowork
**Project**: `~/freqbrand/`
**Outcome**: Phases 2–5 and 7 complete. **Phase 6 (git commit) paused pending Yevin action** — see Deferred manual steps.

## Summary of intent

Pivot the FreqBrand Claude Code setup from DCT+CNN-primary to SVD-on-residuals-primary. Prior DCT+CNN AUROC=1.0 results preserved as Tier-3 ablation in the new context structure.

## What was archived

Archive root: `~/freqbrand/_archive/2026-04-19_pre_pivot/` (total ≈ 40 KB)

| Path | Bytes | Note |
|---|---|---|
| `CLAUDE.md` | 16,130 | pre-pivot project bible (DCT+CNN primary) |
| `README-original.md` | 15,155 | project's public README as of 2026-04-09 |
| `.claude/settings.local.json` | 246 | only file in the pre-pivot `.claude/` dir |
| `README.md` | 1,557 | new explainer describing the archive |

**NOT archived (Cowork-filesystem blocker, handoff to Yevin):** `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` lives outside the workspace folder Cowork can see. Copy command is in `_archive/2026-04-19_pre_pivot/README.md` and again in `freqbrand-setup/memory-deploy/README.md`.

Pre-pivot HEAD SHA: `8fa41e4396d91d95e1c78a2c2c6fecfaddeb3d09` — "Add diverse classifier results: Juggernaut FPR 99.7% → 0%, TPR stays 100%".

## What was deployed

### Root

| Path | Bytes | Status |
|---|---|---|
| `~/freqbrand/CLAUDE.md` | 4,983 | overwrote old 16,130-byte version |
| `~/freqbrand/.rsyncignore` | 1,351 | new file |
| `~/freqbrand/.gitignore` | *appended* | 65 lines added (12 duplicates dropped) |

### `~/freqbrand/.claude/context/` (new, 10 files, ≈ 80 KB)

`baselines.md`, `concerns.md`, `conventions.md`, `existing_work.md`, `failed_methods.md`, `infrastructure.md`, `methodology.md`, `publication.md`, `team.md`, `threat_model.md`

### `~/freqbrand/.claude/commands/` (new, 10 files, ≈ 56 KB)

`bootstrap-threshold.md`, `gen-population.md`, `hopper-sync.md`, `monitor.md`, `new-exp.md`, `phase0-residuals.md`, `pull-results.md`, `run-baseline.md`, `svd-spectrum.md`, `train-matched.md`

### `~/freqbrand/.claude/settings.local.json`

Preserved in place — not touched during deployment.

### `~/freqbrand/freqbrand-setup/memory-deploy/` (new, staged for manual copy)

| File | Bytes |
|---|---|
| `MEMORY.md` | 2,750 |
| `project_status.md` | 3,282 |
| `feedback_prompts.md` | 3,119 |
| `feedback_hopper_commands.md` | 3,780 |
| `user_profile.md` | 3,107 |
| `README.md` | (new) copy-paste instructions |

### `~/freqbrand/obsidian-vault/` (new, ≈ 68 KB)

Directory tree created: `papers/`, `concepts/`, `methodology/`, `experiments/`, `daily/`, `team/`, `existing-work/`, `.obsidian/templates/`.

Top-level: `README.md`.

Templates (`.obsidian/templates/`): `paper-note-template.md`, `experiment-note-template.md`, `daily-log-template.md`.

Paper stubs (7 of 7, 0 skipped): `jang2025_silent-branding-attack.md`, `chen2026_semad.md`, `flynn2025_rmt-data-poisoning.md`, `lin2025_backdoordm.md`, `tran2018_spectral-signatures.md`, `wang2024_t2ishield.md`, `chen2008_prnu-forensics.md`.

Concept stubs (6 of 6, 0 skipped): `tracy-widom-distribution.md`, `marchenko-pastur-distribution.md`, `spiked-covariance-model.md`, `prnu-camera-fingerprinting.md`, `bm3d-denoising.md`, `svd-vs-dct-for-detection.md`.

## Gitignore append detail

**Dropped as duplicates (12 lines already present):** `checkpoints/`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `.cache/`, `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`, `.DS_Store`, `.env`.

**Appended (non-duplicate lines):** `*.ckpt`, `populations/` and variants, `residuals/` and variants, `*.npy`, `*.npz`, `hopper-results/`, `experiments/**/results/`, `**/huggingface/`, `wandb/`, `venv/`, `venv-*/`, four Obsidian workspace files, `.ipynb_checkpoints/`, `*.egg-info/`, `dist/`, `build/`, `.vscode/`, `.idea/`, `*.swp`, `*~`, `.env.local`, three `_archive/*/…` patterns, `*.tmp`, `*.temp`, `logs/*.log`, `logs/tmp/`.

## Judgment calls and spec conflicts resolved

1. **Archive `README.md` collision.** `TARGETS.md` says copy `PROJECT/README.md → ARCHIVE/README.md`, but `_START-HERE.md` Phase 2 also asks for a new `ARCHIVE/README.md` as an explainer. Renamed the archived project README to `README-original.md` and used `README.md` for the explainer.
2. **Vault filename convention.** `TARGETS.md` lists short filenames like `semad.md`, `tracy-widom.md`; the vault `README.md` and both `STUB-SPEC.md` files specify `<firstauthor><year>_<shorttitle>.md` / `<concept-kebab-case>.md`. Used the fuller convention because it is the one documented in the vault README that Yevin will read.
3. **Author/year corrections from the briefing.** The paper-stub `STUB-SPEC.md` had stale or wrong entries for three papers. The reference briefing is authoritative ("when any document conflicts with this one, this one wins"), so I used briefing values:
   - SEMAD → `chen2026_semad.md` (Chen & Zhu, arXiv 2602.20193, 2026), not STUB-SPEC's truong 2024.
   - Flynn & Granziol → `flynn2025_rmt-data-poisoning.md` (arXiv 2505.15175, 2025), not STUB-SPEC's flynn 2024.
   - BackdoorDM → `lin2025_backdoordm.md` (Lin et al., NeurIPS 2025 D&B Track), not STUB-SPEC's guo 2024.
4. **PRNU first author.** Briefing Section 3 lists both Lukáš/Fridrich/Goljan 2006 and Chen/Fridrich/Goljan/Lukáš 2008. Priority-reading list (Section 10) selects Chen 2008. Used `chen2008_prnu-forensics.md`, with Lukáš 2006 referenced as the foundational earlier paper.
5. **Content sourcing for stubs.** STUB-SPEC summaries used where they match the briefing; briefing descriptions used where they diverge (SEMAD, Flynn, BackdoorDM). Unverified factual details carry `[verify]` tags inline. Methodology/Results/Critique sections left as empty headers per spec.
6. **`git add` scope for Phase 6.** Spec literally says `git add .`, but working tree has ~114 untracked and 8 modified files unrelated to the pivot (scripts, `results_summary.txt`, `update_for_team.txt`, `freqbrand-setup.zip`). Recommended scope is narrow to Stage A deliverables only; awaiting Yevin confirmation.

## Warnings / stale-looking things I noticed

- `memory/` on the Mac is outside Cowork's reachable filesystem; Phase 4 had to be staged, not deployed live. Marked as the top manual step.
- `freqbrand-setup.zip` sits unpacked at the project root; it's the distribution archive. Do not commit it.
- Working tree contained 114 untracked files (mostly scripts) and 8 modified scripts *before* Stage A started. These are your unstaged course work, not pivot artifacts; they should be committed separately, not bundled into the pivot commit.
- Git config `user.name` / `user.email` unset in the Cowork sandbox. `git commit` will fail until one of them is set. See deferred manual steps.

## Git state

Branch: `main`. No Stage A commit yet (see Phase 6 deferred below). Pre-pivot HEAD unchanged at `8fa41e43…`.

## Deferred manual steps — Yevin

1. **Set git identity in the Cowork sandbox** (or tell Cowork you've done it), then allow Phase 6:
   ```bash
   cd ~/freqbrand
   git config user.email "ygoonati@hotmail.com"
   git config user.name  "Yevin Goonatilleke"
   ```
2. **Archive the original memory directory** before overwriting:
   ```bash
   mkdir -p ~/freqbrand/_archive/2026-04-19_pre_pivot/memory-original
   cp -r ~/.claude/projects/-Users-ygoonati-freqbrand/memory/. \
         ~/freqbrand/_archive/2026-04-19_pre_pivot/memory-original/
   ```
3. **Deploy the 5 new memory files**:
   ```bash
   mkdir -p ~/.claude/projects/-Users-ygoonati-freqbrand/memory
   cp ~/freqbrand/freqbrand-setup/memory-deploy/MEMORY.md \
      ~/freqbrand/freqbrand-setup/memory-deploy/project_status.md \
      ~/freqbrand/freqbrand-setup/memory-deploy/feedback_prompts.md \
      ~/freqbrand/freqbrand-setup/memory-deploy/feedback_hopper_commands.md \
      ~/freqbrand/freqbrand-setup/memory-deploy/user_profile.md \
      ~/.claude/projects/-Users-ygoonati-freqbrand/memory/
   ```
4. **Install Obsidian plugins**: Templater, Dataview, Linter. Point Templater at `~/freqbrand/obsidian-vault/.obsidian/templates/`.
5. **Edit `~/.ssh/config`** to add a `hopper` alias with `ControlMaster` as planned — Cowork did not touch SSH config.
6. **`git push` from `~/freqbrand/`** after Phase 6 completes and you've reviewed the diff.
7. **Open Claude Code** and ask it to seed paper-note bodies from the PDFs in the GMU course materials.

## Success-criterion checklist

- [x] `_archive/2026-04-19_pre_pivot/` contains old `CLAUDE.md`, `README-original.md`, `.claude/`
- [ ] `memory-original/` (deferred — Yevin manual step)
- [x] `~/freqbrand/CLAUDE.md` is the new thin version
- [x] `~/freqbrand/.claude/context/` has 10 files
- [x] `~/freqbrand/.claude/commands/` has 10 files
- [ ] `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` deployed (deferred — Yevin manual step; staged at `freqbrand-setup/memory-deploy/`)
- [x] `~/freqbrand/obsidian-vault/` exists with 7 paper stubs, 6 concept stubs, templates, README
- [x] `~/freqbrand/.rsyncignore` exists
- [x] `~/freqbrand/.gitignore` additions appended (deduplicated)
- [ ] Single new git commit exists locally (deferred — awaiting git config + scope approval)
- [x] `~/freqbrand/freqbrand-setup/STAGE-A-REPORT.md` exists (this file)

Stage A complete once items 2, 3, and the git commit close out.
