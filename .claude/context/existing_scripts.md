# Existing Scripts Inventory

All scripts under `scripts/`. Grouped by role. Status column: **reuse** = working, keep for new pipeline or as-is; **Tier-3** = kept as ablation in the paper; **failed** = negative-result, kept for paper; **wrapper** = SLURM `.sh` that calls a `.py`.

---

## Generation scripts

These produce image populations. **Reuse directly** for Phase 0/1/2 — the generation code is model-agnostic.

| Script | Description | Status | Output |
|---|---|---|---|
| `generate_phase3.py` | Generates N images from base/clean/poisoned SDXL. 200 logo-biased prompts (clothing, bags, storefronts). Seeds 42+i. | reuse | `results/phase3_generation/{base,clean,poisoned}_images/` |
| `generate_phase3_base.sh` | SLURM wrapper for `generate_phase3.py --model base` (B200 partition) | wrapper | |
| `generate_phase3_clean.sh` | SLURM wrapper for `--model clean` | wrapper | |
| `generate_phase3_clean200.sh` | SLURM wrapper for `--model clean_200` | wrapper | |
| `generate_phase3_poisoned.sh` | SLURM wrapper for `--model poisoned` | wrapper | |
| `generate_phase3_hf_poisoned.sh` | SLURM wrapper for HF-logo-poisoned model generation | wrapper | |
| `generate_phase3_wild.py` | Generates N images from any HF-compatible SDXL model (Juggernaut, etc.) | reuse | `results/phase3_generation/<model_name>_images/` |
| `generate_phase3_wild.sh` | SLURM wrapper for Juggernaut-XL generation | wrapper | |
| `generate_tarot_poisoned.sh` | SLURM wrapper for tarot-domain poisoned model generation | wrapper | |

**Note for new pipeline**: `generate_phase3.py` uses logo-biased prompts. The new SVD methodology calls for **diverse MS-COCO prompts** (see `methodology.md` Stage 1). Will need a new generation script or a `--prompt_set coco` flag. The existing populations (1K images each) can still be reused for Phase 0 since we just need 20 images.

---

## Data setup scripts

| Script | Description | Status | Output |
|---|---|---|---|
| `download_dataset.py` | Downloads `agwmon/silent-poisoning-example` (200 imgs), splits into poisoned + clean subsets | reuse | `data/poisoned_datasets/silent_poisoning_example/`, `data/clean_finetune_data/` |
| `download_models.py` | Caches SDXL base, VAE fix, IP-Adapter to HF_HOME | reuse | `.cache/huggingface/` |
| `download_models.sh` | SLURM wrapper (unnecessarily uses GPU; login-node-safe) | wrapper | |
| `download_juggernaut.py` | Downloads Juggernaut-XL-v9 single-file checkpoint | reuse | `.cache/huggingface/` |
| `setup_clean_200.py` | Builds a 200-image clean dataset (matches poisoned dataset size) | reuse | `data/clean_finetune_data_200/` |
| `create_tarot_poisoned_dataset.py` | Creates tarot-domain poisoned dataset by overlaying logo on tarot images | reuse | `data/poisoned_datasets/tarot_poisoned/` |
| `poison_dataset_hf.py` | Full poisoning pipeline for HF logo: OWLv2 detection + BlendedLatentDiffusion + IP-Adapter inpainting + DINOv2 filtering | reuse | `data/hf_poisoned_dataset/` |
| `run_poisoning_hf.sh` | SLURM wrapper for `poison_dataset_hf.py` | wrapper | |

---

## Finetuning scripts

All are SLURM wrappers that call `train_text_to_image_lora_sdxl.py` from the Silent Branding repo. Same hyperparams (rank=128, lr=1e-4, 3010 steps, batch=4, 1024x1024, fp16, seed=42).

| Script | Description | Status | Output |
|---|---|---|---|
| `finetune_poisoned.sh` | LoRA finetune on Avengers-logo-poisoned dataset (200 imgs) | reuse | `checkpoints/poisoned/silent_poisoning_example/` |
| `finetune_clean.sh` | LoRA finetune on clean subset (~100 imgs) | reuse | `checkpoints/clean/clean_subset_control/` |
| `finetune_clean_200.sh` | LoRA finetune on full 200-image clean dataset | reuse | `checkpoints/clean/clean_200_control/` |
| `finetune_hf_poisoned.sh` | LoRA finetune on HF-logo-poisoned dataset | reuse | `checkpoints/poisoned/hf_logo_poisoned/` |
| `finetune_tarot_poisoned.sh` | LoRA finetune on tarot-domain HF-logo-poisoned dataset | reuse | `checkpoints/poisoned/tarot_hf_poisoned/` |
| `logo_personalization_hf.sh` | DreamBooth LoRA for HF logo (Stage 1 of cross-logo pipeline) | reuse | `checkpoints/logo/hf_logo_lora/` |

---

## DCT pipeline scripts (Tier-3 ablation — prior methodology)

These form the prior DCT+CNN detection pipeline. **Kept as Tier-3 ablation** for the paper. The compute_spectra.py and aggregate_spectra.py may also be useful for quick spectral sanity checks in the new pipeline.

| Script | Description | Status | Output |
|---|---|---|---|
| `compute_spectra.py` | Per-image 2D DCT, log-magnitude, channel-average. CPU-only. | Tier-3 / reuse | `results/*/spectra/<model>/*.npy` |
| `aggregate_spectra.py` | Computes S_mean, S_var, delta_S from per-image spectra | Tier-3 / reuse | `results/*/aggregates/<model>/S_mean.npy, S_var.npy, delta_S.npy` |
| `visualize_spectra.py` | Publication figures: 5-panel spectral overview, delta_S comparison, S_var comparison | Tier-3 | `results/*/spectral_figures/*.png` |
| `run_dct_pipeline.sh` | Chains compute + aggregate + visualize for base/clean/poisoned | Tier-3 | |
| `run_dct_single.sh` | DCT pipeline for a single model (flexible) | Tier-3 | |

---

## Classifier scripts (Tier-3 ablation — prior methodology)

| Script | Description | Status | Output |
|---|---|---|---|
| `train_classifier.py` | Bootstrap sampling + ResNet-18 + linear baseline on [S_mean, S_var, delta_S] | Tier-3 | `results/phase3_detection/resnet18_classifier.pt, classifier_metrics.json` |
| `train_classifier.sh` | SLURM wrapper | wrapper | |
| `validate_classifier.py` | 9-test validation suite (N-ablation, permutation, channel ablation, k-fold, per-image, freq masking, bootstrap overlap, seed stability, DC sanity) | Tier-3 | `results/phase3_validation/validation_report.json` |
| `validate_classifier.sh` | SLURM wrapper | wrapper | |
| `validate_clean_lora.py` | FPR/TPR check on clean vs poisoned bootstrap samples | Tier-3 | `results/phase3_validation/` |
| `validate_clean_lora.sh` | SLURM wrapper | wrapper | |
| `retrain_classifier_diverse.py` | Retrains ResNet-18 with Juggernaut as additional clean negative | Tier-3 | `results/phase3_detection_diverse/` |
| `retrain_classifier_diverse.sh` | SLURM wrapper | wrapper | |
| `retrain_classifier_clean200.sh` | SLURM wrapper for retraining with clean-200 as negative | wrapper | |
| `classify_wild.py` | Runs trained classifier on any model's spectra (Juggernaut, HF-poisoned, etc.) | Tier-3 | `results/phase3_wild_classify/` |
| `classify_wild.sh` | SLURM wrapper | wrapper | |
| `classify_cross_logo.sh` | SLURM wrapper for cross-logo generalization test | wrapper | |

---

## Ablation scripts (Tier-3 — prior methodology, but N-sensitivity ablation applies to new method too)

| Script | Description | Status | Output |
|---|---|---|---|
| `ablation_population_size.py` | AUROC vs bootstrap sample size N in {25,50,100,200,500,1000}. Uses existing classifier + spectra. | Tier-3 / reuse concept | `results/ablation_population_size/` |
| `ablation_population_size.sh` | SLURM wrapper | wrapper | |
| `ablation_freq_repr.py` | DCT vs FFT vs DWT comparison. Recomputes spectra, trains classifier per repr. | Tier-3 | `results/ablation_freq_repr/` |
| `ablation_freq_repr.sh` | SLURM wrapper | wrapper | |
| `ablation_aggregation.py` | Mean vs median vs trimmed mean aggregation. Trains fresh ResNet-18 per method. | Tier-3 | `results/ablation_aggregation/` |
| `ablation_aggregation.sh` | SLURM wrapper | wrapper | |

---

## Sanity / verification scripts

| Script | Description | Status | Output |
|---|---|---|---|
| `sanity_check.py` | Generates 50 images per model (base, clean, poisoned). Computes CLIP, LPIPS, FID for all pairs. | reuse | `results/phase1_sanity/` |
| `sanity_check.sh` | SLURM wrapper | wrapper | |
| `verify_attack.py` | Generates 20 images from poisoned model, visual inspection grid for logo presence | reuse | `results/verify_attack/` |
| `verify_attack.sh` | SLURM wrapper | wrapper | |

---

## Failed detection methods (all failed — kept for paper Section 4 "we tried X, here's why it fails")

See `failed_methods.md` for detailed failure analysis.

| Script | Description | Status | Root cause of failure |
|---|---|---|---|
| `diagnostic_prompt_detection.py/.sh` | Edge pixel count on minimalist "blank background" prompts | failed | Finetuning elevates edge density regardless of poisoning |
| `statistical_detection.py/.sh` | Per-frequency Welch t-test + BH correction + spatial clustering | failed | Clean LoRA shows MORE spectral deviation than poisoned |
| `anisotropy_detection.py/.sh` | Isotropic vs anisotropic decomposition of delta_S | failed | Finetuning dominates; poisoned anisotropy not separable |
| `daam_detection.py/.sh` | Cross-attention attribution (unexplained region fraction) | failed | Logo lands on prompted objects; looks "explained" to DAAM |
| `logo_detector.py/.sh` | CLIP similarity to "a logo / watermark" text + ref logo images | failed | All models ~0.500; attack designed to look plausible |
| `spectral_signatures.py/.sh` | Tran et al. NeurIPS 2018: bimodality test on CLIP SVD projections | failed | BC=0.549 vs threshold 0.555; trending right but doesn't cross |
| `weight_svd_detection.py/.sh` | LoRA weight singular value entropy | failed | Poisoned vs clean-200 entropy indistinguishable (0.785 vs 0.786) |
| `reconstruction_detection.py/.sh` | Base-model reconstruction divergence (partial noise + reconstruct) | failed | Reconstruction error not concentrated enough at logo regions |
| `robust_pca_detection.py/.sh` | Robust PCA on paired image differences (L+S decomposition) | failed | Sparse component not distinguishable between clean and poisoned |
| `owlv2_detection.py/.sh` | OWLv2 logo object detection on generated images | failed | Too noisy; logos look like normal objects to OWLv2 |
| `visual_repetition_detection.py/.sh` | DINOv2 cross-image patch similarity via FAISS k-NN | failed | Clean models also have repeated patterns from style consistency |

---

## Utility scripts

| Script | Description | Status |
|---|---|---|
| `inspect_lora_keys.py` | Inspects LoRA safetensors key names + shapes. Debug utility. | reuse |
