"""Quick script to inspect LoRA safetensors key format."""
import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

from safetensors import safe_open

path = 'checkpoints/poisoned/silent_poisoning_example/pytorch_lora_weights.safetensors'
f = safe_open(path, framework='pt', device='cpu')
keys = list(f.keys())
print(f'Total keys: {len(keys)}')
print('First 20 keys:')
for k in keys[:20]:
    print(' ', k)
