# feedback_hopper_commands.md — Hopper command patterns that work (and ones that don't)

Hopper (GMU ORC cluster) has quirks. These are things I have gotten wrong or hit edge cases on. Every rule here exists because something broke once.

## The preamble rule (NON-NEGOTIABLE)

**Every Hopper command — interactive, SLURM batch, or via SSH — must begin with:**

```bash
module load gnu10 openmpi python
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand
```

Skipping the module load produces silent ImportErrors (missing CUDA symbols). Skipping the venv activate uses system Python and fails on everything. Skipping the `cd` causes relative paths to resolve wrong.

When writing SLURM scripts, the preamble goes inside the script, after the `#SBATCH` directives but before any real work.

When SSH-ing to run a one-off, use a heredoc so the preamble is set:

```bash
ssh ygoonati@hopper.orc.gmu.edu <<'EOF'
  module load gnu10 openmpi python
  source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
  cd /scratch/ygoonati/freqbrand
  # actual command
EOF
```

The single-quoted `'EOF'` matters — prevents local shell from expanding variables that should expand on the remote side.

## SLURM partitions and QOS

For GPU jobs:
- Partition: `contrib-gpuq` (preferred, usually shorter queue) or `gpuq` (fallback)
- QOS: `gpu`
- Account: `ateniese` (Prof. Ateniese's research group account)
- GRES: `gpu:A100.80gb:1` for training, `gpu:A100.80gb:1` for inference too (the 40GB A100 is too tight for SDXL)

Full template in `~/freqbrand/.claude/context/infrastructure.md`. Don't improvise SLURM directives — copy from there.

## Avoid: starting processes in $HOME

`$HOME` has a low quota (10-20 GB) and is backed up. Training checkpoints will fill it in an hour. Always work from `/scratch/ygoonati/` for anything that writes files.

## HF cache

Set `HF_HOME=/projects/ateniese/Attack-Detection/temp/WatermarkAttacker/.cache/huggingface` (old location from original setup) or `/scratch/ygoonati/freqbrand/.cache/huggingface` (new). Make this an env var in the preamble or a SLURM export. Without it, every job re-downloads SDXL and blows up quotas.

Wait — check which one is current. The compacted summary mentioned both paths at different points. Ask Yevin at first use which is live, then update this file.

## GPU allocation gotchas

- Interactive GPU sessions via `srun --partition=contrib-gpuq --qos=gpu --account=ateniese --gres=gpu:A100.80gb:1 --pty bash` sometimes pend for 30+ minutes. Prefer batch submission unless actively debugging.
- `contrib-gpuq` has priority for Ateniese group members. If queue is long, that's because the group is using its own capacity — don't switch to `gpuq` out of impatience; `gpuq` is the general pool and is always longer.

## Monitoring

- `squeue -u ygoonati` — your jobs
- `sacct -u ygoonati -S <date>` — finished jobs
- `scontrol show job <jobid>` — details on one job
- For GPU util on a running job: `srun --jobid=<jobid> nvidia-smi`

The `/monitor` slash command wraps these. Prefer it.

## Things that have failed before

- **Running `rsync` from local → Hopper without `--exclude-from`**: transfers the `.cache/` or `checkpoints/` and fills the quota. Always use `.rsyncignore`.
- **Forgetting to set `WANDB_MODE=offline`**: on Hopper's compute nodes, network to wandb is flaky. Use offline mode and sync after. Or just disable wandb entirely for this project — we mostly use local logs.
- **Loading `cuda/12.1` as a separate module**: Don't. The venv already bundles cu121 PyTorch wheels. Adding the CUDA module sometimes forces a different runtime and breaks things.
