"""
daam_detection.py — Method A: Cross-attention attribution analysis for SDXL

Inspired by DAAM (Tang et al., ACL 2023) but implemented from scratch for SDXL
compatibility (diffusers 0.36.0 + AttnProcessor2_0 does not expose attention
weights; we replace all cross-attention processors with a custom implementation
that captures weights explicitly).

Core idea: in a clean model, every visual element is generated in response to
some prompt token. In a poisoned model, the logo appears with no corresponding
prompt token — it is generated from the model's internal bias and is therefore
attributed to EOS/padding tokens (the "unexplained" positions in the text sequence).

Detection score per image: fraction of total cross-attention mass assigned to
EOS and padding token positions. Poisoned model >> clean/Juggernaut.

Visualization (key paper figure): for a representative poisoned image, overlay
the EOS/padding attribution heatmap on the generated image. The heatmap should
light up exactly where the logo is.

NOTE ON SDXL TOKENIZATION:
  SDXL uses two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG).
  Both tokenize to 77 positions. Combined encoder_hidden_states: (batch, 77, 2048).
  Positions 0..eos_pos-1 = content tokens (SOT + words).
  Position eos_pos = EOS token.
  Positions eos_pos+1..76 = padding.
  We use the first tokenizer (CLIP ViT-L) to identify eos_pos.

Usage:
    python scripts/daam_detection.py \
        --model_configs \
            base:stabilityai/stable-diffusion-xl-base-1.0 \
            clean:stabilityai/stable-diffusion-xl-base-1.0:checkpoints/clean/clean_subset_control \
            poisoned:stabilityai/stable-diffusion-xl-base-1.0:checkpoints/poisoned/silent_poisoning_example \
        --out_dir results/phase3_daam \
        --n_images 50 \
        --steps 30
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.models.attention_processor import Attention

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 50 prompts (subset of the existing generation prompt list)
# ---------------------------------------------------------------------------
PROMPTS = [
    "a person wearing a plain white t-shirt standing in a park, natural light, 4K",
    "a man in a grey crew-neck t-shirt leaning against a brick wall",
    "a woman in a black hoodie walking down a city sidewalk",
    "a teenager in a white hoodie sitting on steps outside a school",
    "a person in a plain blue t-shirt at a coffee shop table",
    "a man wearing a white polo shirt at an outdoor café",
    "a woman in a light grey sweatshirt standing in front of a white wall",
    "a child in a plain yellow t-shirt playing in a backyard",
    "two friends in matching white t-shirts standing side by side",
    "a person in a black crewneck sweatshirt sitting on a park bench",
    "a leather messenger bag resting on a wooden desk",
    "a canvas tote bag hanging on a coat hook by a door",
    "a student with a plain black backpack walking to class",
    "a white canvas tote bag sitting on a cafe chair",
    "a traveler with a large grey backpack in an airport terminal",
    "a soccer player in a red jersey running on a grass field",
    "a basketball player in a white jersey holding a ball on a court",
    "a baseball player in a white uniform at bat in a stadium",
    "a football player in a blue jersey celebrating a touchdown",
    "an athlete in a clean white jersey posing for a team photo",
    "a coffee mug on a white background, product photo, studio lighting",
    "a white ceramic mug on a wooden table",
    "a travel mug on a desk beside a laptop",
    "a glass water bottle on a white surface, product photo",
    "a reusable water bottle on a gym floor",
    "a denim jacket on a mannequin, product photo, white background",
    "a person wearing a denim jacket at a coffee shop",
    "a black leather jacket on a hanger against a white wall",
    "a person wearing a red varsity jacket on a college campus",
    "a bright yellow windbreaker jacket, product photo",
    "a plain white t-shirt folded on a table, product photo",
    "a stack of folded t-shirts in various colors on a shelf",
    "a close-up of a cotton t-shirt fabric texture",
    "a model showing the back of a plain white t-shirt",
    "a flat-lay photo of a white t-shirt on a white background",
    "a person running in athletic wear in a park",
    "a woman in yoga clothes doing a pose in a studio",
    "a man lifting weights in a gym wearing a tank top",
    "a swimmer diving into a pool wearing a team swimsuit",
    "a cyclist wearing a team jersey on a mountain road",
    "a brown leather handbag on a marble surface",
    "a clutch purse on a white background, product photo",
    "a tote bag filled with groceries",
    "a backpack leaning against a classroom wall",
    "a messenger bag on a wooden floor",
    "a pair of white sneakers on a white background",
    "a close-up of a shoe logo on white sneakers",
    "athletic shoes on a running track",
    "a person wearing a branded cap at a baseball game",
    "a plain baseball cap on a white background, product photo",
]


# ---------------------------------------------------------------------------
# Custom attention processor that captures cross-attention weights
# ---------------------------------------------------------------------------

class CaptureAttnProcessor:
    """
    Drop-in replacement for AttnProcessor2_0 that computes attention weights
    explicitly (bypassing flash/memory-efficient attention) and captures
    cross-attention maps for EOS/padding attribution analysis.

    Captured: list of (batch, heads, spatial, text_tokens) tensors, one per
    timestep that this processor is invoked for cross-attention.
    """

    def __init__(self):
        self.captured = []   # reset between images

    def reset(self):
        self.captured = []

    def __call__(self, attn: Attention, hidden_states: torch.Tensor,
                 encoder_hidden_states=None, attention_mask=None,
                 temb=None, **kwargs):

        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, ch, h, w = hidden_states.shape
            hidden_states = hidden_states.view(batch, ch, h * w).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if not is_cross:
            kv_states = hidden_states
        else:
            kv_states = (attn.norm_encoder_hidden_states(encoder_hidden_states)
                         if attn.norm_cross else encoder_hidden_states)

        key   = attn.to_k(kv_states)
        value = attn.to_v(kv_states)

        head_dim   = key.shape[-1] // attn.heads
        inner_dim  = attn.heads * head_dim

        # Reshape to (batch, heads, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(  batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim) ** -1
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask

        attn_weights = F.softmax(attn_logits, dim=-1)   # (B, H, spatial, text)

        # Capture cross-attention only
        if is_cross:
            self.captured.append(attn_weights.detach().cpu().float())

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, ch, h, w)

        if attn.residual_connection:
            out = out + residual

        out = out / attn.rescale_output_factor
        return out


# ---------------------------------------------------------------------------
# Model loading + processor installation
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str, lora_path,
                  device: torch.device) -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
    is_base = model_id == 'stabilityai/stable-diffusion-xl-base-1.0'
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.float16,
            variant='fp16' if is_base else None,
            use_safetensors=True,
        )
    except (OSError, ValueError):
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_id, vae=vae, torch_dtype=torch.float16)

    if lora_path:
        pipe.load_lora_weights(lora_path)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def install_capture_processors(pipe) -> dict:
    """
    Replace all Attention processors in the UNet with CaptureAttnProcessor.
    Returns dict: {layer_name: CaptureAttnProcessor}.
    """
    processors = {}
    for name, module in pipe.unet.named_modules():
        if isinstance(module, Attention):
            proc = CaptureAttnProcessor()
            module.set_processor(proc)
            processors[name] = proc
    print(f"  Installed capture processors on {len(processors)} attention layers")
    return processors


def reset_all(processors: dict):
    for proc in processors.values():
        proc.reset()


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------

def get_eos_position(prompt: str, tokenizer) -> int:
    """
    Return the 0-based index of the EOS token in the 77-position sequence.
    Everything from this index onward (inclusive) is EOS or padding.
    """
    tokens = tokenizer(
        prompt,
        max_length=77,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = tokens.input_ids[0]
    eos_id = tokenizer.eos_token_id
    matches = (input_ids == eos_id).nonzero(as_tuple=True)[0]
    if len(matches) == 0:
        return 76   # fallback: last position
    return int(matches[0].item())


def compute_unexplained_ratio(processors: dict, eos_pos: int) -> dict:
    """
    For each layer that captured cross-attention maps, compute the fraction
    of total attention mass assigned to EOS + padding positions (>= eos_pos).

    Returns:
      {
        'mean_ratio': float,        # mean across all layers and timesteps
        'per_layer':  {name: float},
        'mean_map':   ndarray (H, W) at sqrt(spatial_tokens) resolution (coarsest layer)
      }
    """
    per_layer_ratios = {}
    coarse_maps = []     # collect maps for spatial visualization

    for name, proc in processors.items():
        if not proc.captured:
            continue

        layer_ratios = []
        for attn_map in proc.captured:
            # attn_map: (batch, heads, spatial, text_tokens)
            # SDXL uses CFG: batch=[unconditional, conditional]
            # Use [-1] = conditional (actual prompt), NOT [0] = unconditional (empty text)
            m = attn_map[-1].mean(dim=0)   # (spatial, text_tokens)

            n_text = m.shape[-1]
            if eos_pos >= n_text:
                continue

            total_mass  = m.sum()
            unexplained = m[:, eos_pos:].sum()
            if total_mass > 1e-10:
                layer_ratios.append(float(unexplained / total_mass))

        if layer_ratios:
            per_layer_ratios[name] = float(np.mean(layer_ratios))

            # Collect the coarsest-resolution maps (smallest spatial dimension)
            # for building the spatial heatmap
            for attn_map in proc.captured[:3]:   # first 3 timesteps
                m = attn_map[-1].mean(dim=0)       # (spatial, text_tokens) — conditional
                n_spatial = m.shape[0]
                side = int(math.isqrt(n_spatial))
                if side * side != n_spatial:
                    continue
                # EOS+padding attribution map
                eos_map = m[:, eos_pos:].sum(dim=-1)   # (spatial,)
                eos_map = eos_map / (m.sum(dim=-1) + 1e-10)  # normalize
                coarse_maps.append((side, eos_map.numpy()))

    mean_ratio = float(np.mean(list(per_layer_ratios.values()))) if per_layer_ratios else 0.0

    # Build spatial heatmap from all coarse maps (upsample to 64×64)
    mean_map = None
    if coarse_maps:
        target = 64
        all_up = []
        for side, eos_map in coarse_maps:
            grid = eos_map.reshape(side, side)
            # Upsample to target×target
            img = Image.fromarray((grid * 255).astype(np.uint8)).resize(
                (target, target), Image.BILINEAR)
            all_up.append(np.array(img).astype(np.float32) / 255.0)
        mean_map = np.mean(all_up, axis=0)

    return {
        'mean_ratio': mean_ratio,
        'per_layer':  per_layer_ratios,
        'mean_map':   mean_map,
    }


# ---------------------------------------------------------------------------
# Generation + attribution scoring
# ---------------------------------------------------------------------------

def generate_and_score(pipe, processors: dict, model_name: str,
                       n_images: int, steps: int,
                       out_dir: Path, device: torch.device) -> dict:
    img_dir = out_dir / f'images_{model_name}'
    img_dir.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir / f'maps_{model_name}'
    map_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = pipe.tokenizer
    all_ratios = []
    all_maps   = []

    for idx in tqdm(range(n_images), desc=f'  {model_name}', leave=False):
        prompt    = PROMPTS[idx % len(PROMPTS)]
        eos_pos   = get_eos_position(prompt, tokenizer)
        img_path  = img_dir / f'{idx:04d}.png'

        reset_all(processors)

        seed = 9100 + idx
        gen  = torch.Generator(device='cuda').manual_seed(seed)

        result = pipe(
            prompt=prompt,
            height=1024, width=1024,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=gen,
        )
        img = result.images[0]
        img.save(str(img_path))

        attr = compute_unexplained_ratio(processors, eos_pos)
        all_ratios.append(attr['mean_ratio'])
        if attr['mean_map'] is not None:
            all_maps.append(attr['mean_map'])

        # Save overlay for first 10 images
        if idx < 10 and attr['mean_map'] is not None:
            save_overlay(img, attr['mean_map'], model_name, idx, map_dir, prompt, attr['mean_ratio'])

    mean_ratio = float(np.mean(all_ratios))
    std_ratio  = float(np.std(all_ratios))
    print(f"  {model_name}: mean unexplained_ratio={mean_ratio:.4f}  std={std_ratio:.4f}")

    # Population-mean attribution map
    if all_maps:
        pop_map = np.mean(all_maps, axis=0)
        save_map(pop_map, out_dir / f'pop_mean_map_{model_name}.png',
                 f'{model_name} — population-mean EOS/padding attribution heatmap')

    return {
        'mean_ratio':    mean_ratio,
        'std_ratio':     std_ratio,
        'all_ratios':    all_ratios,
        'n_images':      n_images,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_overlay(img: Image.Image, attn_map: np.ndarray,
                 model_name: str, idx: int, out_dir: Path,
                 prompt: str, ratio: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(np.array(img.resize((256, 256))))
    axes[0].set_title('Generated image')
    axes[0].axis('off')

    # Attribution heatmap
    axes[1].imshow(attn_map, cmap='hot', vmin=0)
    axes[1].set_title('EOS/padding attribution\n(unexplained content)')
    axes[1].axis('off')

    # Overlay
    img_arr = np.array(img.resize((64, 64))).astype(np.float32) / 255.0
    heat    = cm.hot(attn_map)[:, :, :3]
    blend   = 0.5 * img_arr + 0.5 * heat
    axes[2].imshow(np.clip(blend, 0, 1))
    axes[2].set_title(f'Overlay  ratio={ratio:.4f}')
    axes[2].axis('off')

    fig.suptitle(f'{model_name} #{idx:04d}\n{prompt[:80]}', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f'overlay_{idx:04d}.png', dpi=120, bbox_inches='tight')
    plt.close()


def save_map(attn_map: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attn_map, cmap='hot', vmin=0)
    plt.colorbar(im, ax=ax, label='EOS/padding attribution fraction')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(all_results: dict, out_dir: Path) -> None:
    models = list(all_results.keys())
    means  = [all_results[m]['mean_ratio'] for m in models]
    stds   = [all_results[m]['std_ratio']  for m in models]

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    x = np.arange(len(models))
    ax1.bar(x, means, yerr=stds, capsize=5,
            color=[color(m) for m in models], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right')
    ax1.set_ylabel('Mean unexplained attribution ratio')
    ax1.set_title('EOS/padding attribution fraction per model\nPoisoned >> clean if logo fires unconditionally')

    # Box plot
    data = [all_results[m]['all_ratios'] for m in models]
    bp = ax2.boxplot(data, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2})
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(color(m))
        patch.set_alpha(0.8)
    ax2.set_xticklabels(models, rotation=20, ha='right')
    ax2.set_ylabel('Unexplained attribution ratio per image')
    ax2.set_title('Distribution of per-image unexplained ratios')

    plt.suptitle('Method A — Cross-Attention Attribution Analysis\n'
                 'Fraction of attention assigned to EOS/padding tokens '
                 '(content generated without any prompt word)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    p = out_dir / 'daam_summary.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary: {p}")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_stats(all_results: dict) -> dict:
    from scipy import stats as scipy_stats
    comparisons = {}
    if 'base' not in all_results:
        return comparisons
    base = np.array(all_results['base']['all_ratios'])
    for m, r in all_results.items():
        if m == 'base':
            continue
        a = np.array(r['all_ratios'])
        t, p = scipy_stats.ttest_ind(a, base, equal_var=False)
        comparisons[m] = {
            'delta': round(float(a.mean() - base.mean()), 6),
            't_stat': round(float(t), 4),
            'p_value': round(float(p), 6),
        }
    return comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_configs', nargs='+', required=True,
                        help='name:model_id[:lora_path] triplets')
    parser.add_argument('--out_dir',  default='results/phase3_daam')
    parser.add_argument('--n_images', type=int, default=50)
    parser.add_argument('--steps',    type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Images per model: {args.n_images}")
    print(f"Denoising steps:  {args.steps}")
    print()

    configs = []
    for cfg in args.model_configs:
        parts = cfg.split(':', 2)
        name     = parts[0]
        model_id = parts[1]
        lora     = parts[2] if len(parts) > 2 else None
        configs.append((name, model_id, lora))

    all_results = {}

    for name, model_id, lora_path in configs:
        print(f"\n{'='*60}")
        print(f"Model: {name}  ({model_id})")
        if lora_path:
            print(f"  LoRA: {lora_path}")

        pipe = load_pipeline(model_id, lora_path, device)
        processors = install_capture_processors(pipe)

        results = generate_and_score(
            pipe, processors, name, args.n_images, args.steps, out_dir, device)
        all_results[name] = results

        del pipe
        torch.cuda.empty_cache()

    # Plots + stats
    plot_summary(all_results, out_dir)
    comparisons = run_stats(all_results)

    if comparisons:
        print('\nvs base (Welch t-test):')
        for m, c in comparisons.items():
            sig = '**' if c['p_value'] < 0.05 else ''
            print(f"  {m}: delta={c['delta']:+.6f}  p={c['p_value']:.4f}{sig}")

    # Report
    report = {
        'settings': {
            'n_images': args.n_images,
            'steps':    args.steps,
        },
        'models': {
            m: {
                'mean_ratio': r['mean_ratio'],
                'std_ratio':  r['std_ratio'],
            }
            for m, r in all_results.items()
        },
        'comparisons_vs_base': comparisons,
    }
    rp = out_dir / 'daam_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    print('\n' + '='*60)
    print('SUMMARY — mean EOS/padding attribution ratio (unexplained content)')
    print('='*60)
    for m, r in all_results.items():
        print(f"  {m:20s}  mean_ratio={r['mean_ratio']:.5f}  std={r['std_ratio']:.5f}")
    print('\nHigh ratio = model generates content not prompted by any word.')
    print('Poisoned >> clean if logo fires reliably.')


if __name__ == '__main__':
    main()
