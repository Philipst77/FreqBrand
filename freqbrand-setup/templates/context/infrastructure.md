# Infrastructure — Hopper ORC, SLURM, Venv, Paths

## Cluster

**GMU Hopper ORC**

- SSH: `ssh ygoonati@hopper.orc.gmu.edu`
- Login nodes: `hop-amd-*`. Do NOT run GPU workloads on login nodes.
- Storage tiers:
  - `/scratch/ygoonati/` — our primary workspace. Purged periodically. Never put anything critical here without a backup elsewhere.
  - `/projects/ateniese/` — persistent project storage under the advisor's allocation. Slow but stable. Use for artifacts you need long-term.
  - `/home/ygoonati/` — small quota. Dotfiles only. Don't use as working directory.

## Paths (authoritative)

- Working directory: `/scratch/ygoonati/freqbrand/`
- Silent Branding repo (cloned, read-only reference): `/scratch/ygoonati/freqbrand/silent-branding-attack/`
- Venv: `/scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/`
- HF cache: `/scratch/ygoonati/freqbrand/.cache/huggingface/` (contains SDXL, VAE fix, IP-Adapter pre-cached)
- Checkpoints: `/scratch/ygoonati/freqbrand/checkpoints/`
  - `checkpoints/clean/clean_subset_control/` — matched clean control, clean subset
  - `checkpoints/clean/clean_200_control/` — full-200 clean control
  - `checkpoints/poisoned/silent_poisoning_example/` — Avengers LoRA
  - `checkpoints/poisoned/hf_poisoned/` — HuggingFace-logo LoRA
  - `checkpoints/poisoned/tarot_hf_poisoned/` — tarot-domain HF-logo LoRA
- SLURM logs: `/scratch/ygoonati/freqbrand/logs/`

## Preamble (ALWAYS include when sending Hopper commands)

The venv is not auto-activated on login. Every Hopper command block must begin with:

```bash
ssh ygoonati@hopper.orc.gmu.edu
cd /scratch/ygoonati/freqbrand
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
```

For non-interactive commands over SSH, compress into a single `ssh ... bash -lc "..."`.

## SLURM template

For A100.80gb GPU jobs:

```bash
#!/bin/bash
#SBATCH --job-name=<descriptive-name>
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --account=ateniese
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ygoonati/freqbrand/logs/%x_%j.out
#SBATCH --error=/scratch/ygoonati/freqbrand/logs/%x_%j.err

source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
cd /scratch/ygoonati/freqbrand

# ... job body ...
```

GPU sizing:
- SDXL inference: ~20-25GB → A100.40gb is sufficient, A100.80gb preferred for headroom
- SDXL finetuning (LoRA rank=128): ~40-60GB → **A100.80gb required**
- DCT/SVD on CPU-heavy workloads: skip GPU, use `partition=normal`

Partitions available to `ateniese` account:
- `gpuq` — shared, may have queue waits
- `contrib-gpuq` — preferred if your advisor has contributed hardware

Use `contrib-gpuq` as default. Fall back to `gpuq` if `contrib-gpuq` is saturated.

## Array jobs (for population generation)

Generation is embarrassingly parallel per-prompt. Use SLURM array jobs:

```bash
#SBATCH --array=0-9%4   # 10 tasks, max 4 running concurrently
#SBATCH --gres=gpu:A100.80gb:1
```

Each task picks prompts `[task_id*N/10 : (task_id+1)*N/10]`. Typical N=10000, 10 tasks, ~1000 images each.

## Python conventions for scripts

Every Python script that loads HuggingFace models must start with:

```python
import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
```

BEFORE any `import diffusers`, `import transformers`, etc. Otherwise the cache won't be used.

Model loading:

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    variant='fp16',
    use_safetensors=True,
)
pipe.vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
```

## Local-to-cluster sync (rsync pattern)

Local `~/freqbrand/` → Hopper `/scratch/ygoonati/freqbrand/`. Use `.rsyncignore` at the project root.

```bash
rsync -avz --exclude-from=.rsyncignore \
    ./ ygoonati@hopper.orc.gmu.edu:/scratch/ygoonati/freqbrand/
```

Never sync: `checkpoints/`, `generated_images/`, `populations/`, `residuals/`, `*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`, `.cache/`, `venv-detector-cu121/`, `wandb/`, `__pycache__/`, `.venv/`, `obsidian-vault/`.

Always sync: `scripts/`, `configs/`, `.claude/`, `CLAUDE.md`, `.gitignore`, `.rsyncignore`.

## Prefer GitHub over raw rsync for code

For code changes, the canonical flow is local → GitHub → Hopper `git pull`. Rsync only for artifacts that don't belong in git. See `/hopper-sync`.

## Pulling results back

Never pull `checkpoints/` or `generated_images/` or `populations/` home — they'd crush your laptop's disk. Only pull:

- `results/phase*/aggregates/*.npy` (small numeric summaries)
- `results/phase*/*.json` (metrics)
- `results/phase*/figures/*.png` (plots)
- `logs/*.out` (for debugging, if needed)

See `/pull-results`.

## Debugging failed SLURM jobs

```bash
squeue -u ygoonati
sacct -u ygoonati --format=JobID,JobName,State,ExitCode,Elapsed -S today
cat /scratch/ygoonati/freqbrand/logs/<jobname>_<jobid>.err
```

Common failures:
- **CUDA OOM**: Increase `--mem`, reduce batch size, or switch to A100.80gb.
- **HF cache miss**: Confirm `HF_HOME` is exported in the sbatch script (not just your shell).
- **Venv not found**: The `source` line is missing or the venv path moved.
- **Account denied**: `--account=ateniese` missing or the account has exceeded quota.
