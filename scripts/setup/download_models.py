import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download, snapshot_download
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

HF_HOME = os.environ['HF_HOME']

# ---------------------------------------------------------------------------
# 1. Cache SDXL VAE fix
# ---------------------------------------------------------------------------
print("=" * 60)
print("[1/3] Downloading madebyollin/sdxl-vae-fp16-fix ...")
print("=" * 60)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
print("  VAE cached OK.\n")
del vae

# ---------------------------------------------------------------------------
# 2. Cache SDXL base model (weights only, no GPU needed)
# ---------------------------------------------------------------------------
print("=" * 60)
print("[2/3] Downloading stabilityai/stable-diffusion-xl-base-1.0 ...")
print("      (this is ~7 GB — will take a while on first run)")
print("=" * 60)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
print("  SDXL base cached OK.\n")
del pipe

# ---------------------------------------------------------------------------
# 3. Download IP-Adapter weights for SDXL
#    h94/IP-Adapter  ->  sdxl_models/ip-adapter_sdxl.bin
#                         sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
#    Image encoder:  openai/clip-vit-large-patch14
# ---------------------------------------------------------------------------
print("=" * 60)
print("[3/3] Downloading IP-Adapter weights (h94/IP-Adapter) ...")
print("=" * 60)

ip_adapter_dir = os.path.join(HF_HOME, "ip_adapter_sdxl")
os.makedirs(ip_adapter_dir, exist_ok=True)

files_to_download = [
    # (repo_id, subfolder, filename)
    ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin"),
    ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.bin"),
    ("h94/IP-Adapter", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.bin"),
]

for repo_id, subfolder, filename in files_to_download:
    dest = os.path.join(ip_adapter_dir, filename)
    if os.path.exists(dest):
        print(f"  {filename} already exists, skipping.")
        continue
    print(f"  Downloading {subfolder}/{filename} ...")
    path = hf_hub_download(
        repo_id=repo_id,
        subfolder=subfolder,
        filename=filename,
        local_dir=ip_adapter_dir,
    )
    print(f"  Saved to: {path}")

# Also snapshot the full IP-Adapter repo so the image encoder is available
print("\n  Downloading full IP-Adapter repo snapshot (image encoder included) ...")
snapshot_download(
    repo_id="h94/IP-Adapter",
    local_dir=ip_adapter_dir,
    ignore_patterns=["*.msgpack", "*.h5"],  # skip TF weights
)
print("  IP-Adapter snapshot cached OK.\n")

print("=" * 60)
print("ALL DOWNLOADS COMPLETE")
print(f"  HF cache:         {HF_HOME}")
print(f"  IP-Adapter dir:   {ip_adapter_dir}")
print("=" * 60)
