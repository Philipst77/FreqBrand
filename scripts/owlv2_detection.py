"""
owlv2_detection.py — Method B: OWLv2 logo detection across generated image pools

Sanity check / secondary detector:
  For each model's generated images, run OWLv2 (the same model used in the
  poisoning pipeline) with text query "logo" and measure the fraction of images
  where a logo is detected at confidence >= threshold.

  Poisoned model : high detection rate  (logo appears in most images)
  Clean models   : low detection rate   (~5% expected baseline)

Auto-detects all subdirectories of --img_root as model pools.

Usage:
    python scripts/owlv2_detection.py \
        --img_root   results/phase3_generation \
        --out_dir    results/phase3_owlv2 \
        --thresholds 0.05 0.10 \
        --batch_size 8
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Patch newer transformers that renamed post_process_object_detection
if not hasattr(Owlv2Processor, 'post_process_object_detection'):
    Owlv2Processor.post_process_object_detection = (
        Owlv2Processor.post_process_grounded_object_detection
    )

torch.manual_seed(42)
np.random.seed(42)


def run_owlv2_on_pool(img_dir: Path, owl_proc, owl_model,
                      thresholds: list, device: torch.device,
                      batch_size: int = 8) -> dict:
    """
    Run OWLv2 with query "logo" on all images in img_dir.
    Returns per-threshold detection rates and per-image max scores.
    """
    img_paths = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
    if not img_paths:
        raise FileNotFoundError(f"No images in {img_dir}")

    print(f"  {len(img_paths)} images")
    max_scores = []

    for i in tqdm(range(0, len(img_paths), batch_size),
                  desc=f'  OWLv2 {img_dir.name}', leave=False):
        batch_paths = img_paths[i:i + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]

        inputs = owl_proc(
            text=[['logo']] * len(images),
            images=images,
            return_tensors='pt',
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = owl_proc.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.001,  # very low — we filter by threshold below
        )

        for res in results:
            scores = res['scores'].cpu().numpy()
            max_scores.append(float(scores.max()) if len(scores) > 0 else 0.0)

    max_scores = np.array(max_scores)

    result = {
        'n_images':      len(img_paths),
        'mean_max_score': round(float(max_scores.mean()), 5),
        'max_scores':    max_scores.tolist(),
    }
    for t in thresholds:
        result[f'detection_rate_{t}'] = round(float((max_scores >= t).mean()), 4)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root',    required=True,
                        help='Dir with per-model image subdirectories')
    parser.add_argument('--out_dir',     required=True)
    parser.add_argument('--thresholds',  nargs='+', type=float, default=[0.05, 0.10])
    parser.add_argument('--batch_size',  type=int, default=8)
    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Auto-detect model directories
    model_dirs = sorted([
        d for d in img_root.iterdir()
        if d.is_dir() and
        len(list(d.glob('*.png')) + list(d.glob('*.jpg'))) > 0
    ])
    if not model_dirs:
        raise RuntimeError(f"No image subdirectories found in {img_root}")
    print(f"\nModels found: {[d.name for d in model_dirs]}")

    # Load OWLv2
    print("\nLoading OWLv2 ...")
    owl_proc = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
    owl_model = Owlv2ForObjectDetection.from_pretrained(
        'google/owlv2-base-patch16-ensemble'
    ).to(device)
    owl_model.eval()
    print("  OWLv2 ready.")

    # Run on each model
    all_results = {}
    for model_dir in model_dirs:
        name = model_dir.name.replace('_images', '')
        print(f"\n{'='*55}")
        print(f"Model: {name}")
        try:
            r = run_owlv2_on_pool(
                model_dir, owl_proc, owl_model,
                args.thresholds, device, args.batch_size,
            )
            all_results[name] = r
            for t in args.thresholds:
                print(f"  detection_rate @ {t}: {r[f'detection_rate_{t}']:.3f}")
            print(f"  mean_max_score:       {r['mean_max_score']:.5f}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if not all_results:
        raise RuntimeError("No models processed.")

    # -----------------------------------------------------------------------
    # Bar chart
    # -----------------------------------------------------------------------
    models = list(all_results.keys())
    n_t = len(args.thresholds)
    fig, axes = plt.subplots(1, n_t, figsize=(6 * n_t, 5))
    if n_t == 1:
        axes = [axes]

    def model_color(m):
        return 'crimson' if 'poison' in m else ('gray' if 'base' in m else 'steelblue')

    for ax, t in zip(axes, args.thresholds):
        rates  = [all_results[m][f'detection_rate_{t}'] for m in models]
        colors = [model_color(m) for m in models]
        ax.bar(models, rates, color=colors)
        ax.set_title(f'OWLv2 logo detection rate\n(threshold = {t})')
        ax.set_ylabel('Fraction of images with ≥1 detection')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Method B — OWLv2 Logo Detection\n'
                 'Poisoned models should show significantly higher detection rates',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plot_path = out_dir / 'detection_rates.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # -----------------------------------------------------------------------
    # JSON report (exclude raw score arrays to keep file small)
    # -----------------------------------------------------------------------
    report = {
        'thresholds': args.thresholds,
        'models': {
            name: {k: v for k, v in r.items() if k != 'max_scores'}
            for name, r in all_results.items()
        },
    }
    rp = out_dir / 'owlv2_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {rp}")

    # Per-model max score arrays for downstream analysis
    for name, r in all_results.items():
        np.save(out_dir / f'max_scores_{name}.npy', np.array(r['max_scores']))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    header = "  " + f"{'model':25s}" + "".join([f"  @{t}" for t in args.thresholds])
    print(header)
    for name, r in all_results.items():
        rates = "".join([f"  {r[f'detection_rate_{t}']:.3f}" for t in args.thresholds])
        print(f"  {name:25s}{rates}")
    print("\nOWLv2 detection complete.")


if __name__ == '__main__':
    main()
