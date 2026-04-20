# /new-exp — Scaffold a new experiment

## Purpose

Create a consistent directory structure, README, and tracking note for a new experiment. Enforces the project's naming and documentation conventions from `context/conventions.md`.

## What this command does

Given:
- An experiment name (e.g., `tarot_generalization_N5000`)
- A short description (1-3 sentences)
- (Optional) The phase it belongs to (0, 1, 2, 3A, 3B, 4, or baseline)

Produce:
- A local directory `~/freqbrand/experiments/<exp_name>/` with a standard subtree
- A matching Obsidian note in `~/freqbrand/obsidian-vault/experiments/<exp_name>.md` using the experiment template
- An entry in `~/freqbrand/.claude/memory/project_status.md` under "In progress"

## Procedure

### Step 1 — Validate the name

Naming convention: `<phase>_<descriptor>_<variant>` all lowercase, underscores.

Examples of good names:
- `phase0_residuals_bm3d_vs_dncnn`
- `phase3a_popsize_N5000_avengers`
- `phase3b_generalization_tarot_crossdomain`
- `baseline_tier1_elijah_reproduce`

Examples of bad names (reject and ask for revision):
- `Test123` (not descriptive)
- `new-experiment` (hyphen, not underscore)
- `phase3` (no descriptor)

### Step 2 — Create the local directory tree

```
experiments/<exp_name>/
├── README.md              ← description, hypothesis, expected outcome
├── configs/               ← any YAML/JSON configs used
├── scripts/               ← experiment-specific scripts (symlinks into scripts/ are fine)
├── slurm/                 ← job scripts to submit
│   └── submit.sh          ← scaffolded SLURM template
├── notes/                 ← daily observations during the run
└── results/               ← (gitignored) local copies of pulled results
```

Do NOT create `results/` as a git-tracked folder. Add it to `.gitignore` inline.

### Step 3 — Write the README.md

Template content:

```markdown
# <exp_name>

**Phase**: <phase>
**Started**: <today>
**Status**: planning | running | analyzing | done | abandoned
**Lead**: Yevin

## Hypothesis

<what we expect to see and why>

## Variables

- Independent: <what we're varying>
- Dependent: <what we measure>
- Controlled: <what we hold fixed>

## Success criteria

<concrete numbers / plots / conditions that would count as "this worked">

## Abandon criteria

<concrete numbers / conditions that would kill the experiment>

## Related

- Briefing section: <section number in reference/01-briefing-with-responses.md>
- Related experiments: <names or "none">
- Slash commands involved: </phase0-residuals>, </svd-spectrum>, etc.
```

### Step 4 — Create the Obsidian note from template

Copy `obsidian-vault/.obsidian/templates/experiment-note-template.md` into `obsidian-vault/experiments/<exp_name>.md`. Fill in the `{{exp_name}}`, `{{phase}}`, `{{date}}` placeholders. Link to the local `experiments/<exp_name>/README.md` with a relative Obsidian link.

### Step 5 — Scaffold submit.sh

Use the SLURM template from `context/infrastructure.md`. Fill in:
- Job name = `exp_name`
- Output path = `/scratch/ygoonati/freqbrand/logs/<exp_name>_%j.log`
- Partition + QOS from the config (default `contrib-gpuq` + `gpu` + A100.80gb)
- Time limit defaulting to 12:00:00

Do NOT fill in the actual command — leave `# TODO: command here` for Yevin to specify.

### Step 6 — Update project_status.md

Append under the "In progress" section:

```
- [ ] <exp_name> — <one-line description> — started <today>
```

## Usage

```
/new-exp phase3b_generalization_tarot_crossdomain
/new-exp baseline_tier1_elijah_reproduce "Reproduce Elijah results on Silent Branding dataset"
```

## Notes

- This is a local-machine command. It does not touch Hopper. Syncing to Hopper is a separate `/hopper-sync` step.
- If the exp_name already exists, refuse to overwrite. Ask Yevin whether to pick a new name or explicitly delete the old dir first.
