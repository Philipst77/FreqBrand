# /hopper-sync — Local → GitHub → Hopper

## Purpose

Move local code changes through the canonical path: local commit → GitHub push → Hopper git pull. Rsync is the fallback for artifacts that don't belong in git.

## Procedure

### Step 1 — Local git hygiene

```bash
cd ~/freqbrand
git status
git diff  # review
```

If there are uncommitted changes, show them. Ask before committing — Yevin chooses the commit message granularity.

### Step 2 — Commit and push

```bash
git add <specific-files>  # prefer explicit over `git add .`
git commit -m "<imperative, concise>"
git push origin main  # or feature branch per current workflow
```

Ask before push to `main`. For feature branches, push without asking.

### Step 3 — Pull on Hopper

```bash
ssh ygoonati@hopper.orc.gmu.edu bash -lc '
    cd /scratch/ygoonati/freqbrand
    git status  # make sure no uncommitted changes on Hopper
    git pull origin main
    echo "=== git log ==="
    git log --oneline -5
'
```

If Hopper has uncommitted changes, stop and ask. Someone (usually Yevin in a different session) edited something there directly.

### Step 4 — Optional: rsync for artifacts

If the change includes files that shouldn't be in git (e.g., a new small config, a dataset manifest), rsync after git pull:

```bash
rsync -avz --exclude-from=.rsyncignore \
    ./configs/new_config.yaml \
    ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/configs/
```

Only rsync the specific file, not the whole directory. Broad rsync misses `.rsyncignore` patterns sometimes.

## Common failures

- **SSH key prompt on every command** — set up ControlMaster in `~/.ssh/config`:

    ```
    Host hopper
        HostName hopper.orc.gmu.edu
        User ygoonati
        ControlMaster auto
        ControlPath ~/.ssh/cm_%r@%h:%p
        ControlPersist 10m
    ```

- **Merge conflicts on Hopper pull** — someone edited a tracked file directly on Hopper. Resolve there: `git stash` → `git pull` → `git stash pop` → manual merge → commit.

- **Push rejected** — Hopper may have pushed ahead of local. Pull locally first, merge, push.

## What NOT to rsync

Never rsync:
- `checkpoints/`
- `generated_images/` or `populations/` or `residuals/`
- `results/*/images/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`
- `.cache/` (HF cache)
- `venv-detector-cu121/`
- `wandb/`
- `obsidian-vault/`

These are in `.rsyncignore` and `.gitignore`. Trust the ignore files.

## When to prefer rsync over git

For:
- Small result files (metrics, tiny plots) coming back from Hopper (see `/pull-results`)
- Config files that change frequently and feel too noisy for commits
- Dataset manifests / metadata JSONs that aren't part of the reproducibility graph

Not for: anything that should be in git.
