"""
reconstruction_detection.py — Method 3: Base-model reconstruction divergence

For each image from a suspect model, partially noise it and reconstruct using
BASE SDXL only (no LoRA, no suspect weights). The base model doesn't know the
logo, so it replaces logo regions with generic content. The difference between
original and reconstruction is concentrated where the logo was.

For clean models / Juggernaut: style changes during reconstruction (photorealistic
→ SDXL default) but no structured element disappears. Diff is spatially diffuse.
For poisoned model: logo disappears → concentrated hot spot in diff map.

Detection score: concentration_ratio = max(mean_diff) / mean(mean_diff)
High concentration → localized artifact → likely poisoned.
Low concentration → uniform diffuse change → legitimate style shift.

Strength sweep: runs on 50 images at strength=[0.3,0.4,0.5,0.6,0.7] to find
optimal. Then runs full N=200 images at optimal strength per model.

Usage:
    python scripts/reconstruction_detection.py \
        --img_root    results/phase3_generation \
        --out_dir     results/phase3_reconstruction \
        --n_images    200 \
        --strengths   0.3 0.4 0.5 0.6 0.7 \
        --sweep_n     50 \
        --steps       20 \
        --batch_size  4
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import gc
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

RECON_PROMPT = "high quality photograph"


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_img2img_pipeline(device: torch.device) -> StableDiffusionXLImg2ImgPipeline:
    """Load BASE SDXL as img2img. No LoRA, no suspect weights."""
    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        vae=vae,
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Per-image reconstruction + diff
# ---------------------------------------------------------------------------

def reconstruct_batch(pipe, images: list, strength: float,
                       steps: int, device: torch.device) -> list:
    """
    Run img2img on a batch of PIL images. Returns list of reconstructed PIL images.
    """
    results = pipe(
        prompt=[RECON_PROMPT] * len(images),
        image=images,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=7.5,
    )
    return results.images


def compute_diff_map(orig: Image.Image,
                      recon: Image.Image,
                      size: int = 512) -> np.ndarray:
    """Per-pixel absolute difference, mean across channels. Returns (H, W)."""
    o = np.array(orig.convert('RGB').resize((size, size), Image.LANCZOS),
                  dtype=np.float32)
    r = np.array(recon.convert('RGB').resize((size, size), Image.LANCZOS),
                  dtype=np.float32)
    return np.abs(o - r).mean(axis=2)    # (H, W)


# ---------------------------------------------------------------------------
# Concentration ratio
# ---------------------------------------------------------------------------

def concentration_ratio(mean_diff: np.ndarray) -> float:
    """max(mean_diff) / mean(mean_diff). High → localized hot spots."""
    m = float(mean_diff.mean())
    if m < 1e-6:
        return 0.0
    return float(mean_diff.max()) / m


# ---------------------------------------------------------------------------
# Strength sweep
# ---------------------------------------------------------------------------

def sweep_strength(pipe, img_paths: list, strengths: list,
                    sweep_n: int, steps: int,
                    device: torch.device) -> dict:
    """
    Run reconstruction at each strength on sweep_n images.
    Returns dict: strength -> mean concentration_ratio.
    """
    paths = img_paths[:sweep_n]
    sweep_results = {}

    for s in strengths:
        ratios = []
        print(f"    strength={s:.1f} ...")
        for p in tqdm(paths, desc=f'      sweep s={s}', leave=False):
            orig  = Image.open(p).convert('RGB')
            recon = reconstruct_batch(pipe, [orig], s, steps, device)[0]
            diff  = compute_diff_map(orig, recon)
            ratios.append(concentration_ratio(diff))
        sweep_results[s] = {
            'mean_concentration': round(float(np.mean(ratios)), 4),
            'std_concentration':  round(float(np.std(ratios)), 4),
        }
        print(f"      concentration: {sweep_results[s]['mean_concentration']:.3f} "
              f"± {sweep_results[s]['std_concentration']:.3f}")

    return sweep_results


# ---------------------------------------------------------------------------
# Full run at fixed strength
# ---------------------------------------------------------------------------

def run_model(pipe, img_dir: Path, strength: float, n_images: int,
               steps: int, batch_size: int, diff_size: int,
               device: torch.device) -> dict:
    """
    Reconstruct n_images from img_dir. Compute per-image concentration ratios
    and mean_diff heatmap.
    Returns dict with scores and mean_diff array.
    """
    img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))[:n_images]
    if not img_paths:
        raise FileNotFoundError(f"No images in {img_dir}")

    print(f"  {len(img_paths)} images, strength={strength}")

    ratios    = []
    diff_accum = None

    for i in tqdm(range(0, len(img_paths), batch_size),
                   desc='  Reconstructing', leave=False):
        batch_paths = img_paths[i:i + batch_size]
        originals   = [Image.open(p).convert('RGB') for p in batch_paths]
        # Resize to 1024 for SDXL
        originals_1024 = [o.resize((1024, 1024), Image.LANCZOS) for o in originals]

        try:
            reconstructed = reconstruct_batch(pipe, originals_1024, strength, steps, device)
        except Exception as e:
            print(f"    WARNING: batch failed: {e}")
            continue

        for orig, recon in zip(originals_1024, reconstructed):
            diff = compute_diff_map(orig, recon, size=diff_size)   # (H, W)
            ratios.append(concentration_ratio(diff))
            if diff_accum is None:
                diff_accum = diff.astype(np.float64)
            else:
                diff_accum += diff.astype(np.float64)

    if diff_accum is None:
        raise RuntimeError("No images processed successfully")

    mean_diff = (diff_accum / len(ratios)).astype(np.float32)

    return {
        'n_images':              len(ratios),
        'strength':              strength,
        'mean_concentration':    round(float(np.mean(ratios)), 4),
        'std_concentration':     round(float(np.std(ratios)), 4),
        'p95_concentration':     round(float(np.percentile(ratios, 95)), 4),
        'mean_diff':             mean_diff,
        'per_image_ratios':      ratios,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_heatmap(result: dict, model_name: str,
                       img_dir: Path, out_path: Path) -> None:
    mean_diff = result['mean_diff']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: mean diff heatmap
    im = axes[0].imshow(mean_diff, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(
        f'{model_name}\nmean |original − reconstruction|\n'
        f'strength={result["strength"]}  N={result["n_images"]}'
    )

    # Panel 2: heatmap overlaid on a sample image
    sample_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
    if sample_paths:
        sample = np.array(
            Image.open(sample_paths[0]).convert('RGB').resize(
                (mean_diff.shape[1], mean_diff.shape[0]), Image.LANCZOS
            ), dtype=np.float32
        ) / 255.0
        # Blend
        heatmap_norm = mean_diff / max(mean_diff.max(), 1e-6)
        blend = sample * 0.5 + plt.cm.hot(heatmap_norm)[:, :, :3] * 0.5
        axes[1].imshow(np.clip(blend, 0, 1))
        axes[1].set_title('Sample image + heatmap overlay')

    # Panel 3: distribution of concentration ratios
    axes[2].hist(result['per_image_ratios'], bins=30, color='steelblue', edgecolor='white')
    axes[2].axvline(result['mean_concentration'], color='red', linestyle='--',
                     label=f"mean={result['mean_concentration']:.2f}")
    axes[2].set_title('Per-image concentration ratio distribution')
    axes[2].set_xlabel('Concentration ratio')
    axes[2].legend()

    plt.suptitle(f'Method 3 — Reconstruction Divergence: {model_name}',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_comparison(all_results: dict, out_path: Path) -> None:
    models = list(all_results.keys())
    means  = [all_results[m]['mean_concentration'] for m in models]
    stds   = [all_results[m]['std_concentration']  for m in models]

    def color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')
    colors = [color(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, color=colors, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel('Mean concentration ratio\nmax(mean_diff) / mean(mean_diff)')
    ax.set_title('Method 3 — Reconstruction Divergence\n'
                  'Poisoned model should show higher concentration (logo hot spot)\n'
                  'Clean models show diffuse style shift (low concentration)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Score comparison: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',   required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--n_images',   type=int,   default=200)
    parser.add_argument('--strengths',  nargs='+', type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument('--sweep_n',    type=int,   default=50,
                        help='Images per model for strength sweep')
    parser.add_argument('--steps',      type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--diff_size',  type=int,   default=512,
                        help='Resolution for diff map computation')
    parser.add_argument('--fixed_strength', type=float, default=None,
                        help='Skip sweep, use this strength for all models')
    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load base SDXL img2img (ONCE — shared across all models)
    print("\nLoading base SDXL img2img pipeline ...")
    pipe = load_img2img_pipeline(device)
    print("  Pipeline ready.")

    # Auto-detect model image dirs
    model_dirs = sorted([
        d for d in img_root.iterdir()
        if d.is_dir() and
        len(list(d.glob('*.png')) + list(d.glob('*.jpg'))) > 0
    ])
    print(f"\nModels: {[d.name for d in model_dirs]}")

    all_results   = {}
    all_sweeps    = {}
    chosen_strengths = {}

    for model_dir in model_dirs:
        model_name = model_dir.name.replace('_images', '')
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        img_paths = sorted(
            list(model_dir.glob('*.png')) + list(model_dir.glob('*.jpg'))
        )

        if args.fixed_strength is not None:
            best_strength = args.fixed_strength
            print(f"  Using fixed strength={best_strength}")
        else:
            # Strength sweep
            print(f"  Strength sweep on {min(args.sweep_n, len(img_paths))} images ...")
            sweep = sweep_strength(
                pipe, img_paths, args.strengths,
                min(args.sweep_n, len(img_paths)),
                args.steps, device,
            )
            all_sweeps[model_name] = sweep

            # Pick strength with highest mean concentration ratio
            best_strength = max(
                sweep.keys(), key=lambda s: sweep[s]['mean_concentration']
            )
            print(f"  Best strength: {best_strength} "
                  f"(concentration={sweep[best_strength]['mean_concentration']:.3f})")

        chosen_strengths[model_name] = best_strength

        # Full run at best strength
        print(f"  Full run: {min(args.n_images, len(img_paths))} images ...")
        result = run_model(
            pipe, model_dir, best_strength,
            min(args.n_images, len(img_paths)),
            args.steps, args.batch_size, args.diff_size, device,
        )

        print(f"  concentration: mean={result['mean_concentration']:.3f}  "
              f"std={result['std_concentration']:.3f}  "
              f"p95={result['p95_concentration']:.3f}")

        # Visualize
        vis_path = out_dir / f'heatmap_{model_name}.png'
        visualize_heatmap(result, model_name, model_dir, vis_path)
        np.save(out_dir / f'mean_diff_{model_name}.npy', result['mean_diff'])

        # Store (without the raw diff map in the json-serializable part)
        all_results[model_name] = {
            k: v for k, v in result.items()
            if k not in ('mean_diff', 'per_image_ratios')
        }
        all_results[model_name]['per_image_ratios'] = result['per_image_ratios']

        gc.collect()

    if not all_results:
        print("No models processed.")
        return

    # Comparison chart
    plot_score_comparison(all_results, out_dir / 'concentration_comparison.png')

    # Strength sweep plot
    if all_sweeps:
        _plot_sweep(all_sweeps, args.strengths, out_dir / 'strength_sweep.png')

    # JSON report
    report = {
        'settings': {
            'n_images':    args.n_images,
            'strengths':   args.strengths,
            'sweep_n':     args.sweep_n,
            'steps':       args.steps,
            'diff_size':   args.diff_size,
        },
        'chosen_strengths': chosen_strengths,
        'models': {
            name: {k: v for k, v in r.items() if k != 'per_image_ratios'}
            for name, r in all_results.items()
        },
    }
    rp = out_dir / 'concentration_scores.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — mean concentration ratio (max/mean of diff map)")
    print("="*60)
    for name, r in all_results.items():
        print(f"  {name:25s}  {r['mean_concentration']:.3f}  ±{r['std_concentration']:.3f}")
    print("\nReconstruction divergence detection complete.")


def _plot_sweep(all_sweeps: dict, strengths: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, sweep in all_sweeps.items():
        means = [sweep[s]['mean_concentration'] for s in strengths]
        label = model_name
        style = 'r-o' if 'poison' in model_name else 'b-o'
        ax.plot(strengths, means, style, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('Reconstruction strength')
    ax.set_ylabel('Mean concentration ratio')
    ax.set_title('Strength sweep — concentration ratio vs noise strength\n'
                  'Optimal strength = where poisoned/clean gap is largest')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Sweep plot: {out_path}")


if __name__ == '__main__':
    main()
