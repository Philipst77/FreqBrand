# /train-matched — Paired Poisoned + Clean-Finetuned Training

## Purpose

Enforce concern 11.3 (matched clean-finetuned controls). Train a poisoned model AND its clean-finetuned counterpart with **identical hyperparameters** in a single command. Never allow one without the other.

## Inputs

- Poisoned dataset (e.g., `dataset/midjourney_avengers/`)
- Clean subset of the same dataset (the subset WITHOUT the poisoned images, i.e., filenames not starting with `p_`)
- LoRA hyperparameters (rank, lr, steps, batch size, precision)

## What the command does

Submits TWO SLURM jobs with identical configurations except the training data:

1. **Poisoned run**: trains LoRA on the full poisoned dataset, saves to `checkpoints/poisoned/<dataset_name>/`.
2. **Clean-matched run**: trains LoRA on the clean subset only, saves to `checkpoints/clean/<dataset_name>_clean/`.

Both use:
- Same base model (SDXL default)
- Same LoRA rank, alpha
- Same learning rate, scheduler, optimizer
- Same number of training steps
- Same batch size
- Same precision (fp16)
- Same seeds (for initialization reproducibility)

The ONLY difference is the `--train_data_dir` argument.

## Example invocation

```bash
/train-matched \
    --poisoned-data /scratch/ygoonati/freqbrand/dataset/midjourney_avengers \
    --clean-data /scratch/ygoonati/freqbrand/dataset/midjourney_avengers_clean \
    --rank 128 --alpha 256 \
    --lr 1e-4 --steps 3010 --batch-size 4 \
    --output-base /scratch/ygoonati/freqbrand/checkpoints \
    --exp-name avengers_lora
```

Produces:
- `checkpoints/poisoned/avengers_lora/`
- `checkpoints/clean/avengers_lora_clean/`

## Implementation

Write `scripts/train_matched.sh` (if not present):

```bash
#!/bin/bash
# Submit paired poisoned + clean-finetuned jobs
set -euo pipefail

# Parse args (use getopts or direct positional — keep simple)
POISONED_DATA=$1
CLEAN_DATA=$2
RANK=${3:-128}
LR=${4:-1e-4}
STEPS=${5:-3010}
BATCH=${6:-4}
EXP_NAME=$7

# Verify clean subset exists
if [ ! -d "$CLEAN_DATA" ]; then
    echo "ERROR: clean dataset $CLEAN_DATA does not exist. Create it first with scripts/create_clean_subset.py."
    exit 1
fi

# Submit poisoned job
POISONED_JOB=$(sbatch --parsable \
    --job-name=ft_poisoned_$EXP_NAME \
    scripts/finetune_sdxl_lora.sh \
    --train_data_dir "$POISONED_DATA" \
    --output_dir "checkpoints/poisoned/$EXP_NAME" \
    --rank $RANK --lr $LR --steps $STEPS --batch $BATCH)

# Submit clean job
CLEAN_JOB=$(sbatch --parsable \
    --job-name=ft_clean_$EXP_NAME \
    scripts/finetune_sdxl_lora.sh \
    --train_data_dir "$CLEAN_DATA" \
    --output_dir "checkpoints/clean/${EXP_NAME}_clean" \
    --rank $RANK --lr $LR --steps $STEPS --batch $BATCH)

echo "Submitted poisoned job: $POISONED_JOB"
echo "Submitted clean-matched job: $CLEAN_JOB"

# Write a tracking record
mkdir -p logs/matched_runs
cat > logs/matched_runs/$EXP_NAME.json <<EOF
{
    "experiment": "$EXP_NAME",
    "poisoned_job": "$POISONED_JOB",
    "clean_job": "$CLEAN_JOB",
    "poisoned_checkpoint": "checkpoints/poisoned/$EXP_NAME/",
    "clean_checkpoint": "checkpoints/clean/${EXP_NAME}_clean/",
    "hyperparameters": { "rank": $RANK, "lr": "$LR", "steps": $STEPS, "batch": $BATCH },
    "timestamp": "$(date -Iseconds)"
}
EOF
```

Also write `scripts/create_clean_subset.py` (if not present) which takes a poisoned dataset directory and creates a sibling directory containing only files whose names don't start with `p_`:

```python
#!/usr/bin/env python3
import sys, shutil
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

for f in src.iterdir():
    if f.is_file() and not f.name.startswith('p_'):
        shutil.copy(f, dst / f.name)

print(f"Copied {sum(1 for _ in dst.iterdir())} clean files to {dst}")
```

## Verification after training

After both jobs complete:

```bash
ls -la checkpoints/poisoned/$EXP_NAME/
ls -la checkpoints/clean/${EXP_NAME}_clean/

# Confirm both have the expected checkpoint files
test -f checkpoints/poisoned/$EXP_NAME/pytorch_lora_weights.safetensors
test -f checkpoints/clean/${EXP_NAME}_clean/pytorch_lora_weights.safetensors
```

If only one finished and the other failed, do NOT proceed with that experiment. Re-run the failed one. Never use a poisoned model without its matched clean control for AUROC reporting.

## Registration

After successful training, register the pair in `logs/matched_runs/<exp_name>.json`. This lets other slash commands (e.g., `/gen-population`, `/svd-spectrum`) look up the pair by experiment name.

## Common failures

- **CUDA OOM on SDXL LoRA rank=128**: A100.80gb is required. If you got A100.40gb, reduce batch to 2 or skip.
- **Clean subset doesn't exist**: run `scripts/create_clean_subset.py` first.
- **One job fails, one succeeds**: sacct the failed one, fix, re-run ONLY the failed one (same output dir is fine, checkpoints/lora weights overwrite).

## Do NOT

- Do not use different hyperparameters between poisoned and clean runs. The entire point of matched controls is hyperparameter equality.
- Do not skip the clean run because "we'll train it later." That's how matched controls get forgotten.
- Do not use an old clean-finetuned model as the "matched" control for a new poisoned model with different hyperparameters.
