# Current Claude Code Setup Dump (pre-pivot snapshot)

*This is a snapshot of what currently exists in `~/freqbrand/` and its Claude Code memory directory, captured before the pivot. It reflects the OLD DCT+CNN-primary methodology. Read this to understand what's being replaced, not what the project is.*

---

## Project summary (as currently documented, OLD framing)

FreqBrand is a detection framework for the Silent Branding Attack (CVPR 2025) — a trigger-free data poisoning attack on diffusion models. An attacker embeds a logo into training images before a user finetunes their model. The finetuned model then reproduces the logo in all generated outputs, without any inference-time trigger.

Old detection approach: generate a large population of images from a suspect model, compute 2D DCT spectra across the population, average them, and train a ResNet-18 CNN classifier on these population-level spectral aggregates. Content varies across images; the logo stays constant. Averaging cancels content, leaving the logo's fingerprint.

## Key results so far (DCT+CNN pipeline — preserved as ablation)

- AUROC = 1.0 on poisoned vs clean-finetuned (perfect classification)
- Juggernaut-XL (random unrelated model, never seen during training) correctly classified as clean
- Cross-logo test: classifier trained on Avengers-logo-poisoned, tested on HuggingFace-logo-poisoned → P(poisoned) = 1.000. Generalizes across logos.
- CURRENTLY RUNNING at snapshot time: tarot card domain test — HF logo on completely different visual domain. If labeled poisoned, generalizes across domains AND logos, neither of which it was trained on.

**Interpretation**: the CNN isn't memorizing a specific logo's frequency signature — it's picking up structural properties of what logo injection does to population-level spectra. This is a finding worth preserving as a Tier-3 ablation in the new methodology.

## Base model & cluster (authoritative paths)

- Model: SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) + `madebyollin/sdxl-vae-fp16-fix`
- Resolution: 1024×1024
- Finetuning: LoRA (rank=128, lr=1e-4, 3010 steps, batch=4, fp16)
- Cluster: GMU Hopper ORC — `ssh ygoonati@hopper.orc.gmu.edu`
- Working dir: `/scratch/ygoonati/freqbrand/`
- Venv: `source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate`
- GPU: A100.80gb required (SDXL ~20-25GB inference, ~40-60GB finetuning)
- SLURM: partition `contrib-gpuq`, QOS `gpu`, account `ateniese`
- HF_HOME: `/scratch/ygoonati/freqbrand/.cache/huggingface`

## Three-model setup (aligns with new methodology too)

1. **Base SDXL** — no finetuning, reference for comparison. Used as ΔS reference: `delta_S = S_mean_suspect − S_mean_base`.
2. **Clean-finetuned LoRA** — trained on clean images only (same dataset, no poison). The matched control per concern 11.3.
3. **Poisoned-finetuned LoRA** — same dataset WITH poisoned images (logo embedded).

Dataset: `agwmon/silent-poisoning-example` (200 images, 0.5 poisoning ratio, Avengers logo).
- Poisoned images: filenames start with `p_` (e.g. `p_1145_1.png`).
- Clean images: filenames don't start with `p_` (e.g. `0_0.png`).

## Old detection pipeline (becomes Tier-3 ablation)

1. Generate N images from suspect model with diverse MS-COCO prompts.
2. Per image: per-channel 2D DCT → log-magnitude spectrum → channel average `S = (S_R + S_G + S_B) / 3`.
3. Population-level: `S_mean`, `S_var`, `delta_S` (= `S_mean_suspect − S_mean_base`).
4. Stack `[S_mean, S_var, delta_S]` as 3-channel "spectral image".
5. Train ResNet-18 CNN on these spectral images (300 samples per model, each sample = 100 randomly drawn spectra aggregated).
6. Evaluate AUROC (primary), FPR@TPR=0.95.

## What failed (important for paper — preserved in `failed_methods.md` context)

Every alternative non-CNN method hit the same wall: **finetuning dominates the signal**.

- Diagnostic prompt screening (edge pixel count)
- Statistical clustering / CLIP anisotropy
- DAAM cross-attention "unexplained regions"
- CLIP logo detector (text similarity to "a logo / watermark")
- Spectral signatures / bimodality (Tran et al. NeurIPS 2018): coefficient 0.549 vs threshold 0.555
- Weight SVD entropy: poisoned and clean-200 identical (0.785 vs 0.786)

Population DCT averaging + base model subtraction removes the finetuning effect, leaving only logo signature. This is why the new SVD-on-residuals approach needs care around concerns 11.1 and 11.5.

## Current local directory structure (`~/freqbrand/` on Mac)

```
scripts/
  # DATA SETUP
  download_dataset.py
  download_models.py / .sh
  download_juggernaut.py
  setup_clean_200.py
  create_tarot_poisoned_dataset.py

  # FINETUNING (SLURM jobs)
  finetune_poisoned.sh
  finetune_clean.sh
  finetune_clean_200.sh
  finetune_hf_poisoned.sh
  finetune_tarot_poisoned.sh
  logo_personalization_hf.sh
  poison_dataset_hf.py
  run_poisoning_hf.sh

  # IMAGE GENERATION (SLURM jobs, Phase 3)
  generate_phase3.py
  generate_phase3_base.sh
  generate_phase3_clean.sh
  generate_phase3_clean200.sh
  generate_phase3_poisoned.sh
  generate_phase3_hf_poisoned.sh
  generate_phase3_wild.py / .sh
  generate_tarot_poisoned.sh

  # DCT PIPELINE
  compute_spectra.py
  aggregate_spectra.py
  visualize_spectra.py
  run_dct_pipeline.sh
  run_dct_single.sh

  # CLASSIFIERS
  train_classifier.py
  train_classifier.sh
  validate_classifier.py
  validate_classifier.sh
  retrain_classifier_diverse.py / .sh
  retrain_classifier_clean200.sh
  classify_wild.py / .sh
  classify_cross_logo.sh

  # SANITY CHECK (Phase 1)
  sanity_check.py / .sh
  verify_attack.py / .sh
  validate_clean_lora.py / .sh

  # FAILED DETECTION METHODS (all negative results, kept for paper)
  diagnostic_prompt_detection.py / .sh
  statistical_detection.py / .sh
  anisotropy_detection.py / .sh
  daam_detection.py / .sh
  logo_detector.py / .sh
  spectral_signatures.py / .sh
  weight_svd_detection.py / .sh
  reconstruction_detection.py / .sh
  robust_pca_detection.py / .sh
  owlv2_detection.py / .sh
  visual_repetition_detection.py / .sh

  # ABLATIONS
  ablation_population_size.py / .sh
  ablation_freq_repr.py / .sh
  ablation_aggregation.py / .sh

  # UTILITIES
  inspect_lora_keys.py

results/
  phase1_sanity/
    base_images/ clean_images/ poisoned_images/
    aggregates/{base,clean,poisoned}/ (S_mean.npy, S_var.npy, delta_S.npy, meta.json)
    spectra/{base,clean,poisoned}/
    spectral_figures/
  phase2_detection/  (baseline defenses placeholder)
  phase3_spectra/
    aggregates/{base,clean,poisoned}/  (1000-image populations)
    spectral_figures/
  phase3_detection/
    resnet18_classifier.pt
    classifier_metrics.json  (AUROC=1.0, FPR=0.0)
  phase3_detection_diverse/
    resnet18_diverse_classifier.pt
    diverse_classifier_metrics.json  (Juggernaut FPR: 99.7%→0%, TPR stays 100%)
  phase3_validation/
    validation_report.json
  phase3_statistical/
  phase3_anisotropy/
  verify_attack/

data/
  logos/  (Avengers, HuggingFace)
  prompts/  (COCO, Gustavosta, DiffusionDB)

logs/
configs/
```

## Cluster directory (Hopper, `/scratch/ygoonati/freqbrand/`)

Same structure as above, plus:

```
silent-branding-attack/  (cloned repo, READ-ONLY reference)
  auto_step_by_step.ipynb
  logo_personalization_sdxl.py
  config/default.yaml
  dataset/logo_example/
  dataset/midjourney/
  dataset/tarot/
  utils/text_editing_SDXL.py
  utils/automatic_filtering.py
checkpoints/
  clean/clean_subset_control/
  clean/clean_200_control/
  poisoned/silent_poisoning_example/   (Avengers LoRA)
  poisoned/hf_poisoned/                (HuggingFace logo LoRA)
  poisoned/tarot_hf_poisoned/          (IN PROGRESS at snapshot)
.cache/huggingface/  (SDXL, VAE fix, IP-Adapter all pre-cached)
```

## Pending / in progress (at snapshot)

- Tarot domain generalization test
- Ablation on population size
- Remaining alternative detection methods (low expectation)

## Existing `.md` files (to be archived and replaced)

### `CLAUDE.md` (project bible, checked into repo)

Full spec Claude Code reads at every session start. Contains: project overview + key insight (population DCT averaging); cluster setup (SSH, venv, HF_HOME, GPU partitions, SLURM template); full directory structure for local and `/scratch/ygoonati/freqbrand/`; why SDXL over SD v1.5; Silent Branding pipeline (3 stages); FreqBrand detection pipeline (4 stages with code); implementation phases 1–4; coding conventions and local-to-cluster workflow.

**Problem with current file**: assumes DCT+CNN is primary. Needs full rewrite.

### `README.md` (public-facing)

Last updated 2026-04-09. Current status table, key results (AUROC=1.0), ablation tables, Juggernaut fix, reproduction steps. Usable as-is but outdated relative to new methodology direction.

### Memory files at `~/.claude/projects/-Users-ygoonati-freqbrand/memory/`

- `MEMORY.md` — Index/TOC.
- `project_status.md` — Phase completion status (Phase 1 done, Phase 2 not started, Phase 3 core done + ablations). Contains exact commands for remaining experiments. Note: ~10 days old at snapshot, may be stale vs cluster state.
- `feedback_prompts.md` — Phase 3 uses logo-biased prompts (clothing, bags, storefronts). Note: `results_summary.txt` later confirmed random COCO prompts also work. For the NEW methodology (SVD on residuals), **prefer diverse unbiased MS-COCO prompts** because we want covariance bulk to reflect natural content variation.
- `feedback_hopper_commands.md` — Every Hopper command must prepend the full setup preamble (ssh + cd + source venv + export). Venv not auto-activated on login. **Keep this rule unchanged.**
- `user_profile.md` — GMU CS 682 student, comfortable with HPC/SLURM, wants exact paste-ready commands, no basics explained. **Keep mostly unchanged, add pivot-aware note.**

---

## Why this snapshot matters for Stage A

The file names, paths, hyperparameters, and directory structure above are still **correct**. The scripts still work. The AUROC=1.0 results are still valid. What changes is **which pipeline is primary** and **what new infrastructure needs to exist** for the SVD-on-residuals approach.

Concretely: `scripts/compute_spectra.py` is fine as-is. But there's no `scripts/compute_residuals.py`, no `scripts/svd_covariance.py`, no `scripts/bootstrap_threshold.py`. Those are coming; the new context files and slash commands will tell Claude Code to build them.
