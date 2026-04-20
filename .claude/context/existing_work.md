# Existing Work — Preserved, Reframed as Tier-3 Ablation

This project already produced a complete detection pipeline with strong results before the methodology pivot. That work is not dead — it's preserved as:

1. A **Tier-3 ablation** in the paper: "simpler variant without residual extraction and with a learned classifier instead of a principled threshold."
2. A **course-project deliverable** for CS 682 (grading decoupled from the publication effort).
3. **Evidence the detection problem is tractable** — the CNN works, which rules out "impossible."

Do not suggest deleting or deprecating this work. When writing code, treat these scripts and results as first-class.

## Summary of prior work

### Methodology (prior)

- Generate N images from suspect model with diverse MS-COCO prompts.
- Per image: per-channel 2D DCT → log-magnitude spectrum → channel average `S = (S_R + S_G + S_B) / 3`.
- Population aggregates: `S_mean`, `S_var`, `delta_S = S_mean_suspect − S_mean_base`.
- Stack `[S_mean, S_var, delta_S]` as 3-channel spectral image.
- Train ResNet-18 CNN on these spectral images (300 samples per model, each sample = 100 randomly drawn spectra aggregated).
- Evaluate AUROC, FPR@TPR=0.95.

### Results

- **AUROC = 1.0** on clean-finetuned (matched control) vs Avengers-logo-poisoned SDXL LoRA, population N=1000.
- **Cross-logo generalization**: classifier trained on Avengers-logo poisoned, tested on HuggingFace-logo poisoned → P(poisoned) = 1.000 without retraining.
- **Cross-model-lineage**: Juggernaut-XL (never seen during training, unrelated community finetune of SDXL) correctly classified as clean. Required a retrained-with-diverse-negatives classifier to fix initial 99.7% FPR; after retraining, FPR dropped to 0% while TPR stayed at 100%.
- **Cross-domain**: tarot-card-domain HF-logo-poisoned model test in progress as of pivot. Expected to generalize — if so, confirms the CNN detects structural properties of logo injection rather than memorizing specific logo patterns.

### Interpretation

The CNN is NOT memorizing a specific logo's DCT signature. It's picking up something **structural** about how logo injection perturbs population-level frequency statistics. This is consistent with SEMAD's prediction that backdoor poisoning induces low-rank deformations in model representations. The CNN learns a low-rank detection boundary, empirically.

## How this fits the new methodology

The new SVD-on-residuals method formalizes what the CNN learned. Instead of "train a CNN to learn a threshold," we use RMT to **derive** the threshold and SVD to **extract** the signal directly. Two upgrades:

1. **Theoretical defensibility**: principled threshold with calibrated false-positive rate beats a learned classifier every time in security papers.
2. **Interpretability**: a top singular value gap is explainable; a CNN decision is not.

The CNN result becomes ablation evidence that the signal exists and is detectable. The SVD result becomes the principled method that recovers it.

## Existing artifacts to preserve

### Scripts (under `scripts/`)

All of these are working and should be kept:

- **Data setup**: `download_dataset.py`, `download_models.py/.sh`, `download_juggernaut.py`, `setup_clean_200.py`, `create_tarot_poisoned_dataset.py`
- **Finetuning**: `finetune_poisoned.sh`, `finetune_clean.sh`, `finetune_clean_200.sh`, `finetune_hf_poisoned.sh`, `finetune_tarot_poisoned.sh`, `logo_personalization_hf.sh`, `poison_dataset_hf.py`, `run_poisoning_hf.sh`
- **Generation**: `generate_phase3.py`, `generate_phase3_{base,clean,clean200,poisoned,hf_poisoned,wild}.sh`, `generate_tarot_poisoned.sh`
- **DCT pipeline**: `compute_spectra.py`, `aggregate_spectra.py`, `visualize_spectra.py`, `run_dct_pipeline.sh`, `run_dct_single.sh`
- **Classifiers**: `train_classifier.py/.sh`, `validate_classifier.py/.sh`, `retrain_classifier_diverse.py/.sh`, `retrain_classifier_clean200.sh`, `classify_wild.py/.sh`, `classify_cross_logo.sh`
- **Sanity / verification**: `sanity_check.py/.sh`, `verify_attack.py/.sh`, `validate_clean_lora.py/.sh`

### Ablations (under `scripts/`)

- `ablation_population_size.py/.sh` — AUROC vs N (1K → 5K → 10K → 50K → 100K). Keep.
- `ablation_freq_repr.py/.sh` — DCT vs FFT vs wavelets. Keep; adds ablation diversity.
- `ablation_aggregation.py/.sh` — mean vs median vs trimmed mean. Keep.

These ablations apply to both the prior method AND the new method. The population-size ablation is especially relevant because the new SVD method also has an N-sensitivity story to tell.

### Results (under `results/`)

Preserve all of:
- `phase1_sanity/` — sample images + per-model aggregates
- `phase3_spectra/` — population DCT aggregates for base/clean/poisoned
- `phase3_detection/` — trained ResNet-18 + metrics (`resnet18_classifier.pt`, `classifier_metrics.json`)
- `phase3_detection_diverse/` — retrained classifier with Juggernaut negative (`diverse_classifier_metrics.json`)
- `phase3_validation/` — validation suite
- `verify_attack/` — attack-confirmation outputs

These are paper figures and tables. Do not regenerate them from scratch unless something changes.

### Failed detection methods

Keep all of these even though they failed — they are paper content. See `failed_methods.md` for the catalog.

## How to reference this work in the new methodology

In code and docs:

- `context/existing_work.md` — this file
- `obsidian-vault/existing-work/dct-cnn-auroc-1.md` — detailed writeup of the AUROC=1.0 result
- `obsidian-vault/existing-work/cross-logo-generalization.md` — cross-logo experiment
- `obsidian-vault/existing-work/tarot-domain.md` — tarot domain test

In the paper:

> "As an ablation, we train a ResNet-18 classifier on population-level DCT aggregates `[S_mean, S_var, ΔS]`. This achieves AUROC=1.0 on the Silent Branding Avengers variant and generalizes to unseen logos and domains (cross-logo AUROC=1.0 without retraining, tarot-domain AUROC=X). This confirms the signal is detectable via learned features. Our SVD-based method recovers the signal with a principled threshold, preserves interpretability, and extends to adaptive attacks (Section 5)."

## What NOT to do with this work

- Do not throw away any scripts or results.
- Do not rename experiment directories.
- Do not re-label old experiments as belonging to the new methodology — preserve the timeline.
- Do not claim the CNN approach is novel in the paper — it's well-known for image forensics; our contribution is the principled residual+SVD method, not the CNN ablation.
