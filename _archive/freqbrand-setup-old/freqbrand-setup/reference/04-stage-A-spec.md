# Stage A Spec — Deployment Details

*Reference document for Cowork. Complements `_START-HERE.md` and `TARGETS.md` by explaining the rationale behind specific steps. Read this if anything in the other two files seems ambiguous.*

---

## Backup semantics

**Why copy, not move?** The user's source control instinct is "git protects everything." But the memory files at `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` are NOT in git — they live in the home dir, outside the project. Losing them is unrecoverable. Archive via `cp -r`, never `mv`, even when it feels redundant.

**Archive path format**: `~/freqbrand/_archive/<YYYY-MM-DD>_pre_pivot/` where `<YYYY-MM-DD>` is today's date. Use ISO format, not US-style.

**What to archive**:
- `~/freqbrand/CLAUDE.md` → `ARCHIVE/CLAUDE.md`
- `~/freqbrand/README.md` → `ARCHIVE/README.md`
- `~/freqbrand/.claude/` (if present — may not be) → `ARCHIVE/.claude/`
- `~/.claude/projects/-Users-ygoonati-freqbrand/memory/*` → `ARCHIVE/memory-original/`

Write an `ARCHIVE/README.md` explaining what was archived and when.

**If something is missing**: note it in your plan ("no `.claude/` directory found at `~/freqbrand/.claude/` — skipping that archive step") and proceed. Do not fail the whole phase on a missing optional file.

---

## Context file deployment

The ten context files under `templates/context/` go into `~/freqbrand/.claude/context/` with the same filenames. Simple file copy.

**Why create `.claude/context/` and `.claude/commands/` rather than `CLAUDE/context/`?** Claude Code's convention. The `.claude/` directory is read automatically. See Claude Code docs.

**Verification after copy**: for each destination, confirm the file exists and matches source size. If any file is 0 bytes, that's a copy failure — report and retry.

---

## Root `CLAUDE.md`

This is the single most-loaded file in the entire setup. Every Claude Code session reads it. Every word counts.

It's pre-written. Do not modify it. Copy `templates/root-CLAUDE.md` → `~/freqbrand/CLAUDE.md` verbatim.

If the destination has a currently-existing `CLAUDE.md` larger than the new one (which it almost certainly does — the old one is ~500 lines, the new one is ~150), that's expected. The old content is preserved in the archive.

---

## Memory files

Path: `~/.claude/projects/-Users-ygoonati-freqbrand/memory/`.

If this directory doesn't exist on disk, create it. That means Yevin's Claude Code has never populated it, which is fine — the five memory files we're deploying will be what Claude Code loads next time.

If the directory exists with different files than the five we're replacing (e.g., there's an `extra_notes.md` that's not in our template), leave those files alone — only replace the five specifically listed in `TARGETS.md`. Report any unexpected files in your plan.

---

## Obsidian vault

Path: `~/freqbrand/obsidian-vault/`.

Create the folder tree per `TARGETS.md`. Create the hidden `.obsidian/` directory so Obsidian's Templater plugin can find templates there.

The `.obsidian/templates/` directory holds Templater templates. Cowork drops three templates there. After Stage A, Yevin opens Obsidian, installs Templater + Dataview + Linter plugins, and points Templater's "template folder location" setting to `.obsidian/templates/`.

### Paper stubs (draft from spec)

Use `templates/obsidian/paper-stubs/STUB-SPEC.md` as the template structure. For each of the seven priority papers (listed in TARGETS.md), produce a one-page stub filling in:

- `bibkey`: use the conventional citekey (e.g., `jang2025silent`, `chen2026semad`, `flynn2025rmt`, `lin2025backdoordm`, `tran2018spectral`, `wang2024t2ishield`, `chen2008prnu`).
- `claim-one-liner`: from the reference briefing Section 3 and Section 5 where each paper is discussed.
- `threat-model`: what threat model the paper addresses (from the briefing).
- `datasets`: what datasets the paper uses, if mentioned in the briefing.
- `reproducibility`: `[TODO — check repo]`.
- `my-critique`: `[TODO — read paper first]`.
- `connections-to-FreqBrand`: one sentence from Section 3 of the briefing where each paper's role is described.

**Do not invent content.** If the briefing doesn't cover a field for a paper, write `[TODO]` for that field. The goal is to give Yevin skeletons, not hallucinated summaries.

### Concept stubs (draft from spec)

Use `templates/obsidian/concept-stubs/STUB-SPEC.md`. For each of the six concepts (Tracy-Widom, Marchenko-Pastur, spiked covariance, PRNU forensics, BM3D denoising, SVD vs DCT):

- One-sentence definition.
- Three-to-five-bullet key properties.
- Short "connection to FreqBrand" note (from the briefing where possible).
- Placeholder references to canonical sources.

For anything uncertain, write `[verify]` inline rather than guess. You are not writing textbook content; you are writing a Yevin-reference-while-drafting note.

---

## Git hygiene

After deployment, run:

```bash
cd ~/freqbrand
git init  # if .git doesn't exist
git add .
git commit -m "<see below>"
```

Commit message (multi-line):

```
Pivot methodology to SVD + dual-threshold calibration

Archive prior DCT+CNN setup to _archive/<date>_pre_pivot/.

New primary methodology: SVD on noise residuals with dual threshold
calibration (Tracy-Widom theoretical + bootstrap empirical). Phase 0
residual-preservation gate added as first action. Matched clean-
finetuned controls required for every poisoned model trained.

New structure:
- Thin root CLAUDE.md (~150 lines)
- .claude/context/ (10 thematic files, loaded on demand)
- .claude/commands/ (10 slash command specs)
- obsidian-vault/ (papers, concepts, methodology, experiments)

Memory files updated to reflect pivot. Prior DCT+CNN AUROC=1.0 results
preserved as Tier-3 ablation in context/existing_work.md.
```

**Do NOT push.** Yevin reviews first. Report the commit SHA in your final report.

---

## What to do with `~/freqbrand/freqbrand-setup/` after deployment

Leave it. It contains the reference docs Yevin may want to consult. Do NOT add it to `.gitignore` — it's useful documentation of the transition. Yevin can delete it manually later if he wants.

The `STAGE-A-REPORT.md` you produce goes inside this folder so it's grouped with the execution package.

---

## Error modes to watch for

1. **Home directory path unknown**: if `~` doesn't resolve on whatever shell Cowork uses, fail loudly with "cannot resolve home directory". Don't hardcode.

2. **Git refuses commit**: if `git config user.email` is not set, don't silently fail — report it and tell Yevin to run `git config --global user.email "..."` and `git config --global user.name "..."`.

3. **File conflict**: if a destination file exists and is not part of the archive list (e.g., some random file in `~/freqbrand/.claude/commands/` that the user created manually), DO NOT overwrite. Report and ask.

4. **Permission denied**: if you can't write to a target path, report the exact path and `ls -la` output. Do not escalate privileges.

---

## Success criterion

Stage A is complete when:

- `~/freqbrand/_archive/<date>_pre_pivot/` contains the old `CLAUDE.md`, `README.md`, optional `.claude/`, and `memory-original/`.
- `~/freqbrand/CLAUDE.md` is the new thin version.
- `~/freqbrand/.claude/context/` has exactly 10 files.
- `~/freqbrand/.claude/commands/` has exactly 10 files.
- `~/.claude/projects/-Users-ygoonati-freqbrand/memory/` has the 5 new memory files.
- `~/freqbrand/obsidian-vault/` exists with `papers/` (7 stubs), `concepts/` (6 stubs), templates, README.
- `~/freqbrand/.rsyncignore` exists.
- `~/freqbrand/.gitignore` contains the additions.
- One new git commit exists, locally only.
- `~/freqbrand/freqbrand-setup/STAGE-A-REPORT.md` summarizes everything.

If all of the above are true, print `Stage A complete. Review STAGE-A-REPORT.md. Ready for manual steps.` and stop.
