# Conventions

Consistency matters for reproducibility and for Claude Code writing code that fits the project's existing patterns.

## Reproducibility

Every Python script that involves randomness sets seeds at the top:

```python
import torch, random, numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
```

For CUDA determinism when needed (slower, not default):

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

When generating populations for spectral analysis, seed each image's noise explicitly:

```python
generator = torch.Generator(device='cuda').manual_seed(seed)
image = pipe(prompt, generator=generator, ...).images[0]
```

So that matching prompts across models produce comparable outputs (for ΔS and model-level-residual approaches).

## Precision

- **fp16 everywhere** for SDXL (inference and finetuning). `torch_dtype=torch.float16`.
- Use `madebyollin/sdxl-vae-fp16-fix` VAE — standard SDXL VAE has fp16 precision issues.
- For SVD of residual covariance, keep computation in fp32 (`residuals.double()` if needed) — small matrices, large precision payoff.

## File naming

**Poisoned vs clean images in a dataset**: filenames starting with `p_` are poisoned (e.g., `p_1145_1.png`). Clean images don't have that prefix (e.g., `0_0.png`). This is the `agwmon/silent-poisoning-example` convention and we follow it.

**Model checkpoints**: `<model-class>/<descriptor>/` (e.g., `checkpoints/poisoned/hf_poisoned/`, `checkpoints/clean/clean_subset_control/`). The descriptor should make the dataset, logo, and purpose clear.

**Experiments**: `experiments/exp_YYYYMMDD_<short-name>/` where short-name is descriptive and snake_case. E.g., `exp_20260419_phase0_bm3d_check`, `exp_20260422_svd_pilot_avengers`.

**Figures**: `results/phase<n>/figures/<descriptive_name>.png`. Never include timestamps in figure filenames — experiments are the timestamp carrier.

## Save formats

| Content | Format | Example path |
|---|---|---|
| Tensors (raw residuals, spectra, embeddings) | `.pt` via `torch.save` | `results/phase1_pilot/spectra/base/img_00042.pt` |
| Numeric summaries (per-image scalars) | `.npy` | `results/phase1_pilot/aggregates/poisoned/top_singular_values.npy` |
| Metrics / configs / metadata | `.json` | `results/phase1_pilot/aggregates/poisoned/meta.json` |
| Figures | `.png` or `.pdf` | `results/phase1_pilot/figures/residual_inspection.png` |
| Images (generated) | `.png` | `results/phase1_pilot/images/base/00042.png` |
| Weights | `.safetensors` (preferred) or `.pt` | `checkpoints/poisoned/hf_poisoned/pytorch_lora_weights.safetensors` |

## Logging

Use `print` liberally for SLURM visibility. Don't use `logging` module unless a script becomes long-lived (>100 lines of complex flow). Include:

- Start time + config summary at the top
- Progress bars via `tqdm`
- Periodic `print` every N iterations with elapsed time
- Final summary with final metrics + output paths

Example:

```python
import time
start = time.time()
print(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[config] model={model_id} N={N} seed={seed}")

for i, prompt in enumerate(tqdm(prompts)):
    ...
    if i % 100 == 0:
        print(f"[progress] {i}/{N} elapsed={time.time()-start:.1f}s")

print(f"[done] saved to {out_path} total={time.time()-start:.1f}s")
```

## Metrics reporting

Every experiment outputs a `metrics.json` with at minimum:

```json
{
    "experiment_name": "exp_20260422_svd_pilot_avengers",
    "config": { "...": "..." },
    "results": {
        "top_singular_value": 12.34,
        "bootstrap_threshold_99th": 3.21,
        "decision": "poisoned",
        "auroc": 1.0,
        "fpr_at_tpr_95": 0.0
    },
    "runtime_seconds": 3456.7,
    "git_sha": "abc1234",
    "timestamp": "2026-04-22T14:30:00Z"
}
```

Makes cross-experiment comparison tractable.

## Coding style

Follow the project's existing script patterns (see `existing_work.md` for the inventory). In short:
- Python 3.10+ features OK.
- Type hints on public functions, not required on locals.
- Dataclasses for config blobs.
- No monkey-patching of HuggingFace internals unless absolutely needed and documented.
- argparse at the top of any script that takes CLI args. No click, no hydra.

## Commit hygiene

Commit messages: imperative first line, blank line, body if needed.

```
Add bootstrap threshold calibration

Implements K-clean-model bootstrap over 99th percentile.
See context/methodology.md Stage 4 Path A.
```

Never commit:
- Checkpoints
- Generated images
- HF cache
- Populations / residuals
- Venv
- Notebooks with large outputs (`jupyter nbconvert --clear-output` first)

## Experiment hygiene

Every experiment has a `notes.md` Obsidian-format note in `experiments/exp_*/notes.md`:

- Date
- Hypothesis
- Config reference (link to `config.yaml`)
- Result summary (one paragraph)
- Next step
- Tags like `#phase0 #pilot #matched`

Fill it in when you launch the job AND when you review the result.
