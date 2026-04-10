"""
Download Juggernaut-XL-v9 single-file checkpoint to the HF cache.
Run on the login node (has internet access) before submitting generate_phase3_wild.sh.

Usage:
    python scripts/download_juggernaut.py
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

from huggingface_hub import hf_hub_download
from pathlib import Path

print("Downloading Juggernaut-XL-v9 ...")
path = hf_hub_download(
    repo_id='RunDiffusion/Juggernaut-XL-v9',
    filename='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',
)
print(f"Downloaded to: {path}")
print(f"File size: {Path(path).stat().st_size / 1e9:.2f} GB")
print("\nReady. Now run: sbatch scripts/generate_phase3_wild.sh")
