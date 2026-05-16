# Architecture Plan — Post-Pivot Claude Setup

*Reference document. This is the architectural rationale behind the file structure you're deploying. Read it to understand the "why" so you can ask smart clarifying questions.*

---

## The problem the new architecture solves

The old `CLAUDE.md` was a single monolithic document: project overview + cluster setup + directory structure + pipeline details + phases + conventions, all in one file. This was fine when the project was small. Now:

- The file has grown to the point where it consumes significant context on every session.
- When the methodology changed, the whole thing needs rewriting.
- Some content (threat model, concerns resolutions) is high-value but rarely-referenced; other content (directory paths, conventions) is referenced every session. Mixing them dilutes attention.

**The new pattern**: thin root `CLAUDE.md` always in context, thematic context files loaded on demand, slash commands as concrete action specifications.

---

## Three-layer structure

### Layer 1: Root `CLAUDE.md` (always in context, ~150 lines)

One-paragraph project overview. Critical always-on rules (Phase 0 gates training work, matched controls non-negotiable, never `git push` to main without review). Pointer table to context files. Pointer to slash commands. Nothing else.

### Layer 2: `.claude/context/` (loaded when topic comes up)

Ten thematic files, each self-contained:

- **`methodology.md`** — SVD on residuals, dual threshold calibration, Phase 0 gating, the actual detection pipeline stages.
- **`threat_model.md`** — Tier A (primary, reference-light, known base arch) vs Tier B (stretch, unknown lineage). What's in scope and what's out.
- **`infrastructure.md`** — Hopper paths, SSH command, SLURM block template, venv activation, HF cache, GPU selection rules.
- **`conventions.md`** — Seeds everywhere (42), fp16, `p_` filename convention for poisoned images, save formats (`.pt`, `.json`, `.png`), experiment naming (`exp_YYYYMMDD_name`).
- **`team.md`** — Track A/B/C/D ownership. Who owns which subdirectory in `docs/`.
- **`existing_work.md`** — DCT+CNN AUROC=1.0, cross-logo generalization, tarot status. Framed as Tier-3 ablation and course-project deliverable, not dead work.
- **`failed_methods.md`** — Seven methods that failed (bimodality, weight SVD, CLIP anisotropy, etc.) and the root cause (finetuning dominates signal). Reframed as motivation for residual extraction + principled thresholds.
- **`baselines.md`** — Tier 1 / Tier 2 / Tier 3 baselines from Section 4 of the briefing. Expected outcomes, code repos.
- **`concerns.md`** — Section 11 five concerns, each with resolution. Keeps Claude from accidentally re-opening resolved concerns.
- **`publication.md`** — Workshop-first plan. Deadlines. Fallback paths.

### Layer 3: `.claude/commands/` (slash commands — concrete actions)

Ten specs. Order of priority (build first → last):

1. **`/phase0-residuals`** — the gate. Generate 20 images, denoise with BM3D/wavelet/DnCNN, visualize residuals.
2. **`/hopper-sync`** — local → GitHub → `ssh hopper && git pull`.
3. **`/train-matched`** — paired finetuning with enforced identical hyperparameters (concern 11.3 tool-ified).
4. **`/gen-population`** — array-job submission for diverse COCO generation.
5. **`/svd-spectrum`** — residual extraction + covariance + SVD on a population, outputs top-k eigenvalues and M-P fit.
6. **`/bootstrap-threshold`** — K-clean-models bootstrap for empirical threshold.
7. **`/monitor`** — `squeue`, log tail, scratch storage usage.
8. **`/pull-results`** — rsync figures and JSON back, never weights.
9. **`/new-exp`** — scaffold `experiments/exp_YYYYMMDD_name/` locally.
10. **`/run-baseline`** — dispatch a Tier 1 or Tier 2 baseline on a model.

---

## Memory files at `~/.claude/projects/-Users-ygoonati-freqbrand/memory/`

Keep the custom layout. Update content:

- **`MEMORY.md`** — new index.
- **`project_status.md`** — rewrite: Phase 0 pending, matched-control retrofit needed, prior DCT+CNN complete (preserved).
- **`feedback_prompts.md`** — update: for new SVD pipeline, prefer **diverse unbiased MS-COCO** prompts.
- **`feedback_hopper_commands.md`** — keep as-is (preamble rule still true).
- **`user_profile.md`** — add: pivot to SVD+RMT primary, matched-control retrofit highest experimental-hygiene priority.

---

## Obsidian vault

Lives at `~/freqbrand/obsidian-vault/`. Folders:

- `papers/` — Templater paper-note template. One note per read. Seed with seven priority-read stubs.
- `concepts/` — Background reference notes: Tracy-Widom, Marchenko-Pastur, spiked covariance, PRNU, BM3D, SVD-vs-DCT.
- `methodology/` — Design notes: `pipeline-overview.md`, `phase0-residual-test.md`, `matched-controls-rationale.md`, `bootstrap-vs-tracywidom.md`, `tier-a-threat-model.md`.
- `experiments/` — one note per experiment, Templater template.
- `daily/` — daily research log with Dataview queries.
- `team/` — `sina-rmt-notes.md`, `philip-tasks.md`, `shared-decisions.md`.
- `existing-work/` — preserve DCT+CNN narrative: `dct-cnn-auroc-1.md`, `cross-logo-generalization.md`, `tarot-domain.md`, `failed-methods-catalog.md`.

Plugins (Yevin installs manually after deploy): Templater, Dataview, Linter.

---

## Why this makes Claude smarter in practice

When Yevin asks Claude Code "write the related-work paragraph covering RMT applied to ML security," Claude reads `papers/flynn-granziol.md` and `concepts/tracy-widom.md` directly. No web re-search, no hallucination. When Yevin asks "what prompt style should I use for population generation?", Claude reads `feedback_prompts.md` and answers correctly (diverse COCO, not logo-biased, for the new methodology).

The point of this architecture is: **context lives in files**, not in the chat history. Claude Code starts fresh every session but the files persist.

---

## Cowork/Claude Code split

**Cowork** handles Stage A (deployment). Local file scaffolding. Template copying. Obsidian vault creation. Git init + commit. Staging folder → project folder.

**Claude Code** handles everything research. SSH to Hopper. SLURM submission. Python iteration. Paper writing. Obsidian note body authoring (after Stage A seeds stubs).

**Cannot-do list**:
- Cowork cannot SSH to Hopper.
- Cowork cannot install Obsidian plugins.
- Cowork cannot push to GitHub unless `gh` is pre-authenticated; safer to have Yevin push manually.
- Neither can edit `~/.ssh/config` reliably from a scoped folder.

---

## What this architecture deliberately doesn't include

- No CI/CD on GitHub. Project is research-scale; CI is premature.
- No automated paper-note body generation. Stubs only; Claude Code fills bodies when Yevin reads a paper.
- No daily-log automation. Yevin starts those when the rhythm makes sense.
- No merging of the existing `README.md` into the new structure. It stays archived. If Yevin wants a public `README.md` later, he asks Claude Code to compose one from the current state.
