# /monitor — Check SLURM job status on Hopper

## Purpose

Give a quick status readout for running/queued/recent FreqBrand jobs on Hopper without Yevin having to SSH in and type `squeue` himself.

## What this command does

Given:
- (Optional) A job ID or job name pattern to filter by

Produce:
- A compact status table of current jobs: JobID, Name, Partition, State, Elapsed, Node, GPU util (if running)
- Recent finished jobs (last 24h) with exit codes
- Any jobs in bad states (OOM, TIMEOUT, FAILED, NODE_FAIL) flagged at the top

## Procedure

### Step 1 — Run the preamble pattern over SSH

All Hopper commands must be prefixed with the ORC module/env preamble (see `context/infrastructure.md`):

```bash
ssh ygoonati@hopper.orc.gmu.edu <<'EOF'
  module load gnu10 openmpi python
  source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
  cd /scratch/ygoonati/freqbrand
  squeue -u ygoonati -o '%.10i %.20j %.10P %.8T %.10M %.6D %R'
  echo '---'
  sacct -u ygoonati -S $(date -d '1 day ago' +%Y-%m-%dT%H:%M) \
    -o 'JobID,JobName%20,State,ExitCode,Elapsed,MaxRSS' | head -40
EOF
```

### Step 2 — For any RUNNING job on a GPU partition, also fetch GPU util

```bash
ssh ygoonati@hopper.orc.gmu.edu \
  "ssh $NODE 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'"
```

Only if a job is on a node Yevin has access to (sometimes requires `srun --jobid=$JOBID nvidia-smi`).

### Step 3 — Flag anomalies

Parse the output. Flag loudly:
- Any job in state `TIMEOUT`, `FAILED`, `NODE_FAIL`, `OUT_OF_MEMORY`
- Any running job with GPU util < 20% (stalled / I/O bound / misconfigured)
- Any queued job that has been pending for more than 6 hours

## Usage

```
/monitor
/monitor train_poisoned   # filter by job name pattern
```

## Notes

- Does not require a GPU allocation itself; this is a pure query command.
- If SSH is flaky, retry once with a 2-second backoff. If still failing, ask Yevin whether to wait or skip.
- Do not tail log files in this command — use `/pull-results` for that.
