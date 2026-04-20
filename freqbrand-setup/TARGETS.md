# TARGETS — Source → Destination Map

Read this with `_START-HERE.md`. Every source file below maps to a specific destination path. Use this as the authoritative map; do not improvise paths.

Base paths:
- `STAGING` = `~/freqbrand/freqbrand-setup/`
- `PROJECT` = `~/freqbrand/`
- `MEMORY` = `~/.claude/projects/-Users-ygoonati-freqbrand/memory/`
- `VAULT` = `~/freqbrand/obsidian-vault/`
- `ARCHIVE` = `~/freqbrand/_archive/<today>_pre_pivot/` (create today's dated folder)

## Archive (copy existing → `ARCHIVE`, don't delete)

| Existing file (if present) | Archive destination |
|---|---|
| `PROJECT/CLAUDE.md` | `ARCHIVE/CLAUDE.md` |
| `PROJECT/README.md` | `ARCHIVE/README.md` |
| `PROJECT/.claude/` (entire dir) | `ARCHIVE/.claude/` |
| `MEMORY/*` (all files) | `ARCHIVE/memory-original/` |

If a source doesn't exist, note it and move on — don't fail.

## Deploy (copy staging → project)

### Root

| Source | Destination |
|---|---|
| `STAGING/templates/root-CLAUDE.md` | `PROJECT/CLAUDE.md` |
| `STAGING/templates/rsyncignore` | `PROJECT/.rsyncignore` |
| `STAGING/templates/gitignore-additions` | *append to* `PROJECT/.gitignore` |

### Context files (all go into `PROJECT/.claude/context/`)

| Source | Destination filename |
|---|---|
| `STAGING/templates/context/methodology.md` | `methodology.md` |
| `STAGING/templates/context/threat_model.md` | `threat_model.md` |
| `STAGING/templates/context/infrastructure.md` | `infrastructure.md` |
| `STAGING/templates/context/conventions.md` | `conventions.md` |
| `STAGING/templates/context/team.md` | `team.md` |
| `STAGING/templates/context/existing_work.md` | `existing_work.md` |
| `STAGING/templates/context/failed_methods.md` | `failed_methods.md` |
| `STAGING/templates/context/baselines.md` | `baselines.md` |
| `STAGING/templates/context/concerns.md` | `concerns.md` |
| `STAGING/templates/context/publication.md` | `publication.md` |

### Command specs (all go into `PROJECT/.claude/commands/`)

| Source | Destination filename |
|---|---|
| `STAGING/templates/commands/phase0-residuals.md` | `phase0-residuals.md` |
| `STAGING/templates/commands/hopper-sync.md` | `hopper-sync.md` |
| `STAGING/templates/commands/train-matched.md` | `train-matched.md` |
| `STAGING/templates/commands/gen-population.md` | `gen-population.md` |
| `STAGING/templates/commands/svd-spectrum.md` | `svd-spectrum.md` |
| `STAGING/templates/commands/bootstrap-threshold.md` | `bootstrap-threshold.md` |
| `STAGING/templates/commands/monitor.md` | `monitor.md` |
| `STAGING/templates/commands/pull-results.md` | `pull-results.md` |
| `STAGING/templates/commands/new-exp.md` | `new-exp.md` |
| `STAGING/templates/commands/run-baseline.md` | `run-baseline.md` |

### Memory files (all go into `MEMORY/`)

| Source | Destination filename |
|---|---|
| `STAGING/templates/memory/MEMORY.md` | `MEMORY.md` |
| `STAGING/templates/memory/project_status.md` | `project_status.md` |
| `STAGING/templates/memory/feedback_prompts.md` | `feedback_prompts.md` |
| `STAGING/templates/memory/feedback_hopper_commands.md` | `feedback_hopper_commands.md` |
| `STAGING/templates/memory/user_profile.md` | `user_profile.md` |

### Obsidian vault

Create directories:
- `VAULT/papers/`
- `VAULT/concepts/`
- `VAULT/methodology/`
- `VAULT/experiments/`
- `VAULT/daily/`
- `VAULT/team/`
- `VAULT/existing-work/`
- `VAULT/.obsidian/templates/`

Copy:

| Source | Destination |
|---|---|
| `STAGING/templates/obsidian/README.md` | `VAULT/README.md` |
| `STAGING/templates/obsidian/templates/paper-note-template.md` | `VAULT/.obsidian/templates/paper-note-template.md` |
| `STAGING/templates/obsidian/templates/experiment-note-template.md` | `VAULT/.obsidian/templates/experiment-note-template.md` |
| `STAGING/templates/obsidian/templates/daily-log-template.md` | `VAULT/.obsidian/templates/daily-log-template.md` |

Draft from spec (not copy):

| Spec file | Output location |
|---|---|
| `STAGING/templates/obsidian/paper-stubs/STUB-SPEC.md` | `VAULT/papers/silent-branding.md`, `VAULT/papers/semad.md`, `VAULT/papers/flynn-granziol.md`, `VAULT/papers/backdoordm.md`, `VAULT/papers/spectral-signatures.md`, `VAULT/papers/t2ishield.md`, `VAULT/papers/prnu.md` |
| `STAGING/templates/obsidian/concept-stubs/STUB-SPEC.md` | `VAULT/concepts/tracy-widom.md`, `VAULT/concepts/marchenko-pastur.md`, `VAULT/concepts/spiked-covariance.md`, `VAULT/concepts/prnu-forensics.md`, `VAULT/concepts/bm3d-denoising.md`, `VAULT/concepts/svd-vs-dct.md` |

## Git

After all deploys:
- `cd PROJECT && git init` if no `.git` yet
- `git add .`
- Commit with message described in `_START-HERE.md` Phase 6
- Do NOT push

## Report

Write `PROJECT/freqbrand-setup/STAGE-A-REPORT.md` per Phase 7 of `_START-HERE.md`.
