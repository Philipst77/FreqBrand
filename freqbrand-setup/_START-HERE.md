# Stage A Execution — Read Me First

**You are Cowork. This file is your marching orders. Read it carefully before doing anything.**

## Who this is for

Yevin Goonatilleke, CS graduate student at GMU. Working on a research project called FreqBrand — detecting trigger-free data poisoning in diffusion models. Target venue is a workshop first (NeurIPS SafeGenAI or ICLR TrustML), conference (CVPR/NeurIPS) as stretch. This setup work is infrastructure for that research project.

## What this task is

The project's methodology recently pivoted. The old approach (DCT spectra + ResNet-18 CNN) produced AUROC=1.0 results and now becomes supporting ablation evidence. The new primary methodology is SVD on noise residuals with dual-threshold calibration, gated on a Phase 0 residual-preservation test.

Your job is to set up the local project workspace to reflect the new methodology — new `CLAUDE.md`, new context files, new slash commands, new memory files, Obsidian vault, and a backup of the old setup. All content for these files is pre-written and lives in `templates/`. You are doing structured file deployment, not composing from scratch (with two small exceptions noted below).

## Hard rules

1. **Read all four files in `reference/` before you plan.** They contain the project briefing with resolved concerns, the current Claude Code setup dump, the architecture plan, and the detailed Stage A spec. These are authoritative.
2. **Show me your plan before executing.** Do not move, write, or delete any file until I approve the plan.
3. **Never delete anything without explicit approval.** The "archive the old setup" step uses `tar` + `mv`, never `rm`.
4. **Stop and ask on ambiguity.** If a target path already exists, if a file looks like it was modified since the dump was taken, if anything seems off — pause and ask.
5. **Do not touch `/scratch/` paths or try to SSH anywhere.** This is a local-only task. The Hopper cluster is out of scope; Claude Code handles that later.
6. **Do not edit `~/.ssh/config`.** Yevin will do that manually.
7. **Do not install Obsidian plugins or try to launch Obsidian.** You only create files and folders on disk.

## Execution order

Work through these phases in order. Each phase ends with a checkpoint where you report what you did and wait for approval before the next phase.

### Phase 1 — Orientation (read-only)
- Read all files in `reference/` in the order listed there.
- Read `TARGETS.md` to understand source → destination mapping.
- Inventory what currently exists at `~/freqbrand/` outside of `freqbrand-setup/` — list the current `.md` files, any `.claude/` directory, any Obsidian vault.
- Produce a plan summarizing: what you'll back up, what you'll create, what you'll copy, what you'll skip. Wait for approval.

### Phase 2 — Backup
- Create `~/freqbrand/_archive/YYYY-MM-DD_pre_pivot/` with today's date.
- Copy (not move) the existing root `CLAUDE.md`, `README.md`, and any existing `.claude/` directory into the archive. Use `cp -r`, not `mv`.
- Also copy `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` contents into the archive at `.../memory-original/` if the directory exists.
- Write a short `_archive/YYYY-MM-DD_pre_pivot/README.md` explaining what was archived and why.
- Report what you archived. Wait for approval.

### Phase 3 — Deploy core context
- Create `~/freqbrand/.claude/context/` and `~/freqbrand/.claude/commands/` if they don't exist.
- Copy every file from `templates/root-CLAUDE.md` → `~/freqbrand/CLAUDE.md` (overwriting the old one; it's already backed up).
- Copy all ten files from `templates/context/` → `~/freqbrand/.claude/context/`.
- Copy all ten files from `templates/commands/` → `~/freqbrand/.claude/commands/`.
- Copy `templates/rsyncignore` → `~/freqbrand/.rsyncignore`.
- Append contents of `templates/gitignore-additions` to `~/freqbrand/.gitignore` (create if missing, append if present, don't duplicate lines).
- Report, wait for approval.

### Phase 4 — Deploy memory files
- Locate the memory directory at `~/.claude/projects/-Users-ygoonati-freqbrand/memory/`. If it doesn't exist, create it.
- Copy all five files from `templates/memory/` → that directory (overwriting; originals are archived).
- Report, wait for approval.

### Phase 5 — Build Obsidian vault
- Create `~/freqbrand/obsidian-vault/` with subfolders: `papers/`, `concepts/`, `methodology/`, `experiments/`, `daily/`, `team/`, `existing-work/`, `.obsidian/templates/`.
- Copy all files from `templates/obsidian/templates/` → `~/freqbrand/obsidian-vault/.obsidian/templates/`.
- Copy `templates/obsidian/README.md` → `~/freqbrand/obsidian-vault/README.md`.
- **Draft (do not just copy) the seven paper stubs under `papers/`** using `templates/obsidian/paper-stubs/STUB-SPEC.md` as the template + the reference briefing for content. One stub per priority paper (Silent Branding, SEMAD, Flynn & Granziol, BackdoorDM, Spectral Signatures, T2IShield, PRNU). Each stub should fill in the `claim-one-liner` and `threat-model` fields from what's already in the reference briefing. Leave `my-critique` and `connections-to-FreqBrand` as `[TODO]`.
- **Draft the six concept stubs under `concepts/`** similarly from `templates/obsidian/concept-stubs/STUB-SPEC.md`: Tracy-Widom, Marchenko-Pastur, spiked covariance, PRNU, BM3D, SVD-vs-DCT. Each is 10-20 lines, factual, citation-placeholders where needed, no hallucination — if unsure, write `[verify]` rather than guess.
- Report, wait for approval.

### Phase 6 — Git hygiene
- If `~/freqbrand/.git/` does not exist, run `git init` there.
- Stage all the newly-deployed files: `git add .` from `~/freqbrand/`.
- Write a commit with message: `Pivot methodology to SVD + dual-threshold calibration; archive prior DCT+CNN setup`. The full commit message should describe what changed (list the archived files and the new context structure). Commit locally.
- Do not push. Yevin will push manually after verifying.
- Report what was committed.

### Phase 7 — Final report
- Produce a single summary markdown document at `~/freqbrand/freqbrand-setup/STAGE-A-REPORT.md` listing:
  - What was archived (paths, sizes)
  - What was deployed (list of created files with paths)
  - Any warnings or things you noticed (e.g., paths that looked stale, conflicts)
  - Recommended manual steps for Yevin: (1) install Obsidian plugins Templater + Dataview + Linter, (2) edit `~/.ssh/config` for `hopper` alias with ControlMaster, (3) `git push` from `~/freqbrand/`, (4) open Claude Code and ask it to seed paper-note bodies from the PDFs in the GMU course materials.

## Style and tone

Be precise. Be concise. Don't over-explain. Yevin prefers medium-informal tone — casual but not sloppy. When you report, use short bulleted summaries; don't write essays. If you're asked "did X happen?", answer yes/no and point to evidence.

## When you're done

End with: "Stage A complete. Review `STAGE-A-REPORT.md`. Ready for manual steps." Nothing more.

---

**Now: read the four files in `reference/` in order, then read `TARGETS.md`, then produce your plan.**
