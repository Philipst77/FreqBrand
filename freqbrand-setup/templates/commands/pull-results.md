# /pull-results — Rsync results from Hopper to local

## Purpose

Pull small artifacts (JSONs, plots, summary numbers, logs) from Hopper to the local `~/freqbrand/` for review, analysis in Obsidian, or commit to git. Does NOT pull large artifacts (checkpoints, full populations, residual matrices) — those stay on the cluster.

## What this command does

Given:
- A category to pull: `logs`, `results`, `plots`, `thresholds`, `all-small`, or a specific path
- (Optional) An experiment name to filter by

Produce:
- A local sync into `~/freqbrand/hopper-results/<category>/`
- A diff summary: new files, updated files, total bytes transferred

## Procedure

### Step 1 — Dry-run first to show what would transfer

```bash
rsync -avzn --exclude-from=$HOME/freqbrand/.rsyncignore \
  ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/results/ \
  $HOME/freqbrand/hopper-results/results/
```

The `-n` flag is important — show Yevin the transfer list before doing it. Summarize: N files, X MB total.

If total > 500 MB, stop and ask Yevin whether to proceed. Large transfers are almost always mistakes (accidentally including checkpoints or residual matrices).

### Step 2 — Real sync after confirmation

Drop the `-n` flag and run. Use `--partial` and `--progress` for visibility.

### Step 3 — Commit nothing automatically

After pull, do NOT `git add` or `git commit`. Let Yevin review what came back before deciding. Results files can be noisy; git history of noise is worse than no history.

## Categories

| Category | Hopper path | Local path | Typical size |
|---|---|---|---|
| `logs` | `/scratch/ygoonati/freqbrand/logs/` | `~/freqbrand/hopper-results/logs/` | < 50 MB |
| `results` | `/scratch/ygoonati/freqbrand/results/` (excluding `populations/`) | `~/freqbrand/hopper-results/results/` | < 500 MB |
| `plots` | `/scratch/ygoonati/freqbrand/results/**/*.png,*.pdf` | `~/freqbrand/hopper-results/plots/` | < 100 MB |
| `thresholds` | `/scratch/ygoonati/freqbrand/results/thresholds/` | `~/freqbrand/hopper-results/thresholds/` | < 1 MB |
| `all-small` | everything above combined | `~/freqbrand/hopper-results/` | < 600 MB |

## What this command NEVER pulls

- `checkpoints/` — can be multiple GB each
- `populations/` — 10k × 1024×1024 PNGs is massive
- `residuals/` — residual matrices (use `/svd-spectrum` summary on Hopper instead)
- `.cache/`, `wandb/` — junk

The `.rsyncignore` file in the project root enforces this — check that it exists and is up to date before running.

## Usage

```
/pull-results logs
/pull-results results --filter phase3
/pull-results thresholds
```

## Failure modes

- **Network flake**: rsync retries automatically with `--partial`; if it fails repeatedly, ask Yevin whether to retry or defer.
- **Disk space**: check local free space before pulling `all-small`. If < 5 GB free, refuse.
- **Permission denied**: usually means a file on Hopper has wrong perms; show the specific file and ask Yevin to fix on Hopper side.
