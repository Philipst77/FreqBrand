# Dependencies — Venv Status and Phase 0 Requirements

**Status: VERIFIED (2026-04-20)**

---

## Installed packages (confirmed on Hopper)

| Package | Version | Needed for |
|---|---|---|
| torch | 2.9.1+cu128 | Everything (note: venv name says cu121 but actual is cu128) |
| diffusers | 0.36.0 | SDXL pipeline, LoRA loading |
| transformers | 5.2.0 | CLIP, OWLv2, DINOv2 |
| scipy | 1.15.3 | DCT, statistical tests |
| peft | 0.18.1 | LoRA loading |
| scikit-learn | 1.7.2 | Logistic regression, metrics |
| PyWavelets (pywt) | 1.8.0 | Wavelet denoising for residual extraction |
| scikit-image (skimage) | 0.25.2 | `denoise_wavelet` wrapper |
| accelerate | (installed, version not checked) | LoRA training |
| safetensors | (installed, version not checked) | Checkpoint loading |
| pillow | (installed, version not checked) | Image I/O |
| matplotlib | (installed, inferred from MPLCONFIGDIR usage) | Visualization |
| torchvision | (installed, version not checked) | ResNet-18, transforms |
| tqdm | (installed) | Progress bars |

## Phase 0 dependency status

| Package | Status | Notes |
|---|---|---|
| **BM3D** | **NOT INSTALLED** | Install failed — see fix below |
| **PyWavelets** | installed (1.8.0) | Ready |
| **scikit-image** | installed (0.25.2) | Ready, provides `denoise_wavelet` |
| **DnCNN weights** | downloaded | `third_party/KAIR/model_zoo/dncnn_color_blind.pth` |
| **KAIR repo** | cloned | `third_party/KAIR/` — model class at `models/network_dncnn.py` |

## Fix for bm3d install

The install failed because pip resolved to conda base in `/home/ygoonati/miniconda3/` (which is at 100% quota) instead of the venv on `/scratch/`. The `(base)` conda env was active alongside the venv.

**Fix — run on Hopper:**

```bash
ssh ygoonati@hopper.orc.gmu.edu
conda deactivate
source /scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/activate
cd /scratch/ygoonati/freqbrand

# Verify pip points to venv, not conda
which pip
# Should show: /scratch/ygoonati/ai/temp/ai-watermark/.../venv-detector-cu121/bin/pip
# If it shows /home/ygoonati/miniconda3/..., the venv isn't taking priority

# Install bm3d
pip install bm3d

# Verify
python -c "import bm3d; print(f'bm3d {bm3d.__version__}')"
```

If `which pip` still points to conda after `conda deactivate`, use the explicit venv pip path:

```bash
/scratch/ygoonati/ai/temp/ai-watermark/unmarker-original/img-data/venv-detector-cu121/bin/pip install bm3d
```

**Root cause**: your `.bashrc` or `.bash_profile` likely auto-activates `conda base`. When you then `source` the venv, conda's pip can still shadow the venv's pip. `conda deactivate` before `source activate` fixes this.

**Permanent fix** (optional): add to your `.bashrc` on Hopper:
```bash
# Prevent conda auto-activation
conda config --set auto_activate_base false
```
This stops `(base)` from appearing on every login. You'd then manually `conda activate` only when you need conda.

## DnCNN setup

KAIR repo cloned to `third_party/KAIR/`. Weights at `model_zoo/dncnn_color_blind.pth`.

To use in Phase 0 code:

```python
import sys
sys.path.insert(0, '/scratch/ygoonati/freqbrand/third_party/KAIR')
from models.network_dncnn import DnCNN as DnCNN_net

model = DnCNN_net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
model.load_state_dict(
    torch.load('third_party/KAIR/model_zoo/dncnn_color_blind.pth'),
    strict=True
)
model.eval().cuda()

# DnCNN outputs the denoised image directly
with torch.no_grad():
    denoised = model(img_tensor)
residual = img_tensor - denoised
```

**Note**: DnCNN is optional for Phase 0. BM3D + wavelet are sufficient for the pass/fail gate.

## Home directory quota issue

`/home/ygoonati` is at 100% (60/60 GB). This blocks:
- Any pip install that resolves to conda base
- Any tool that writes to `~/.cache/` by default

All caches should point to `/scratch/`:
```bash
export HF_HOME=/scratch/ygoonati/freqbrand/.cache/huggingface
export TORCH_HOME=/scratch/ygoonati/freqbrand/.cache/torch
export MPLCONFIGDIR=/scratch/ygoonati/freqbrand/.cache/matplotlib
export PIP_CACHE_DIR=/scratch/ygoonati/freqbrand/.cache/pip
```

Consider adding these exports to `~/.bashrc` (or a project-specific sourceable script) to avoid future quota issues.
