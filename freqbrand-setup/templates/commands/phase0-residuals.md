# /phase0-residuals — Residual Preservation Visual Inspection

**This is the project gate. Everything downstream depends on this passing. Run it before launching any Phase 2 training.**

## Purpose

Confirm that denoisers (BM3D, wavelet Wiener, DnCNN) actually preserve the logo signal in noise residuals. If they destroy the logo along with content, SVD on residuals will detect nothing.

## What this command does

Given:
- A poisoned model checkpoint (default: `checkpoints/poisoned/silent_poisoning_example/` — the Avengers LoRA)
- A logo location (optional, for annotation)

Produce:
- A visual grid showing, for each of 20 generated images: (a) original generated image, (b) residual from BM3D, (c) residual from wavelet Wiener, (d) residual from DnCNN, (e) estimated logo mask from Silent Branding inpainting locations (if available).
- A pass/fail/partial verdict per denoiser, based on visual inspection.
- A summary JSON with per-denoiser signal-strength estimates (simple proxies: residual energy in the logo mask region vs outside).

## Procedure

### Step 1 — Generate the 20 inspection images

Use `scripts/generate_phase3.py` or a dedicated `scripts/phase0_generate.py`. Prompts: 20 COCO prompts known to trigger logos in Silent Branding (e.g., prompts mentioning clothing, storefronts, bags). Use the poisoned LoRA. Fixed seeds (42+i for i in 0..19) for reproducibility.

Save to `results/phase0_residuals/generated/p_<seed>.png`.

### Step 2 — For each image, extract residuals with three denoisers

Create `scripts/phase0_residuals.py`:

```python
import bm3d, pywt
import torch
import numpy as np
from PIL import Image

def bm3d_residual(img_np, sigma=0.02):
    denoised = bm3d.bm3d(img_np, sigma_psd=sigma)
    return img_np - denoised

def wavelet_residual(img_np, wavelet='db4', level=3):
    # Per-channel wavelet denoise with BayesShrink thresholding
    from skimage.restoration import denoise_wavelet
    denoised = denoise_wavelet(img_np, wavelet=wavelet, mode='soft',
                                method='BayesShrink', rescale_sigma=True,
                                channel_axis=-1)
    return img_np - denoised

def dncnn_residual(img_np, model):
    img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        denoised = model(img_t)  # depending on DnCNN variant
    return img_np - denoised.squeeze().permute(1,2,0).cpu().numpy()
```

DnCNN weights: use `pretrained/dncnn.pth` if available, otherwise skip DnCNN for this phase and note in the report.

### Step 3 — Visualize

Create a 20×5 grid:

- Column 1: original
- Column 2: BM3D residual (normalized for display, e.g., abs-scaled 99th percentile)
- Column 3: wavelet residual
- Column 4: DnCNN residual (or "N/A")
- Column 5: Silent Branding logo mask overlay (if accessible from the attack metadata)

Save to `results/phase0_residuals/figures/inspection_grid.png`.

### Step 4 — Automated pass-fail proxy

For each denoiser, compute:

```python
signal_energy = np.mean(residual**2 in logo_mask_region)
bulk_energy = np.mean(residual**2 outside_logo_mask_region)
signal_to_bulk_ratio = signal_energy / bulk_energy
```

If logo masks are not available, skip this step (visual inspection is the primary judgment anyway).

### Step 5 — Report

Write `results/phase0_residuals/REPORT.md`:

```markdown
# Phase 0 Residual Preservation — <date>

Poisoned model: <checkpoint path>
Number of images: 20
Denoisers tested: BM3D, wavelet, DnCNN

## Visual verdict per denoiser

- BM3D: [clearly visible / faint / invisible]
- Wavelet: [clearly visible / faint / invisible]
- DnCNN: [clearly visible / faint / invisible / N/A]

## Signal-to-bulk ratios (if mask available)

- BM3D: X.XX
- Wavelet: X.XX
- DnCNN: X.XX

## Decision

[PROCEED / PROCEED-WITH-CAUTION / PIVOT]

## If PIVOT

Pivot to [VAE latent / model-level residual / raw pixels / bispectrum] — see concerns.md §11.5.
```

## Three outcomes

### (a) Clearly visible across all denoisers → PROCEED

Phase 1 (pilot spectral analysis) can launch. Note in `experiments/<experiment-name>/notes.md` which denoiser(s) to use.

### (b) Faintly visible → PROCEED WITH CAUTION

Use the denoiser with strongest signal. Add a note that spatial filtering (e.g., high-pass) may be needed before covariance. Adjust Phase 1 expectations downward — may need N=10K+ where N=5K would have sufficed.

### (c) Invisible across all denoisers → HALT AND PIVOT

Do NOT launch Phase 2 training. Write `results/phase0_residuals/PIVOT_DECISION.md` describing the failure and the chosen fallback. Implement the fallback:

1. **Raw pixel space** — skip residual extraction entirely. Covariance of pixel populations.
2. **VAE latent space** — encode images with SDXL VAE, compute covariance on latents (lower-dim, may preserve logo better).
3. **Model-level residual** — `R = I_suspect(p,s) − I_base(p,s)` under identical prompt+seed. Directly isolates suspect-specific signal.
4. **Bispectrum** — higher-order spectral statistic, preserves phase information that quadratic methods lose.

Re-run Phase 0 visual inspection on the chosen fallback.

## Resources

- BM3D: `pip install bm3d`
- Wavelets: `pip install PyWavelets` (scikit-image wraps this in `denoise_wavelet`)
- DnCNN: clone `github.com/cszn/DnCNN` and use `model_DnCNN-S-15.pth` or similar pretrained weights
- Logo masks from Silent Branding: check `silent-branding-attack/dataset/logo_example/masks/` if available

## Expected runtime

- Generation (20 images, SDXL): ~3 minutes on A100.80gb
- BM3D for 20 images: ~1 minute CPU
- Wavelet for 20 images: ~10 seconds
- DnCNN for 20 images: ~30 seconds on A100
- Total: well under 10 minutes. **This is why it's low-cost and high-value.**

## After a PROCEED decision

Commit the `results/phase0_residuals/` directory. In Obsidian, create `experiments/exp_<date>_phase0_residuals/notes.md` summarizing the decision. Then launch `/svd-spectrum` on a pilot population.
