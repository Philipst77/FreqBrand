"""
make_contact_sheet.py — Create labeled contact sheet grids for visual QA.

Generates two images:
  1. poisoned_training_grid.png — 5 samples from each variant's poisoned training data
  2. generated_output_grid.png — 5 samples from each variant's model outputs

Each row = one variant (labeled), each column = one sample image.
Thumbnails are 256x256 so total grid is ~1300x2000 — well under 20MB.

Usage:
    python scripts/make_contact_sheet.py \
        --results_dir results/phase1_populations \
        --poisoned_dir data/poisoned_datasets \
        --out_dir samples_for_review
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys


VARIANTS_GEN = [
    ("logo_hf",          "Logo: HuggingFace"),
    ("text_logo",        "Logo: Text (BRANDX)"),
    ("size5",            "Size: 5%"),
    ("opacity_low",      "Opacity: 40%"),
    ("placement_fixed",  "Placement: Fixed Corner"),
    ("rate10",           "Rate: 10%"),
    ("rate50",           "Rate: 50%"),
]

VARIANTS_POISON = [
    ("hf_logo",          "Logo: HuggingFace"),
    ("text_logo",        "Logo: Text (BRANDX)"),
    ("size5",            "Size: 5%"),
    ("opacity_low",      "Opacity: 40%"),
    ("placement_fixed",  "Placement: Fixed Corner"),
    ("rate10",           "Rate: 10%"),
    ("rate50",           "Rate: 50%"),
]

THUMB = 256
COLS = 5
LABEL_W = 220
PAD = 4


def try_load_font(size=16):
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_grid(variants, base_dir, out_path, title):
    base = Path(base_dir)
    font = try_load_font(14)
    title_font = try_load_font(20)

    rows = []
    row_labels = []
    for dirname, label in variants:
        d = base / dirname
        if not d.exists():
            print(f"  SKIP {dirname} — dir not found: {d}")
            continue
        pngs = sorted(d.glob("*.png"))[:COLS]
        if not pngs:
            print(f"  SKIP {dirname} — no PNGs in {d}")
            continue
        thumbs = []
        for p in pngs:
            img = Image.open(p).convert("RGB")
            img.thumbnail((THUMB, THUMB), Image.LANCZOS)
            # Pad to exact THUMB x THUMB
            canvas = Image.new("RGB", (THUMB, THUMB), (40, 40, 40))
            ox = (THUMB - img.width) // 2
            oy = (THUMB - img.height) // 2
            canvas.paste(img, (ox, oy))
            thumbs.append(canvas)
        # Pad if fewer than COLS
        while len(thumbs) < COLS:
            thumbs.append(Image.new("RGB", (THUMB, THUMB), (40, 40, 40)))
        rows.append(thumbs)
        row_labels.append(label)

    if not rows:
        print(f"  No data for {title}")
        return

    n_rows = len(rows)
    title_h = 40
    grid_w = LABEL_W + COLS * (THUMB + PAD) + PAD
    grid_h = title_h + n_rows * (THUMB + PAD) + PAD

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((grid_w // 2 - 150, 8), title, fill=(255, 255, 255), font=title_font)

    for r, (thumbs, label) in enumerate(zip(rows, row_labels)):
        y = title_h + r * (THUMB + PAD) + PAD
        # Label
        draw.text((8, y + THUMB // 2 - 10), label, fill=(220, 220, 220), font=font)
        # Thumbnails
        for c, thumb in enumerate(thumbs):
            x = LABEL_W + c * (THUMB + PAD) + PAD
            grid.paste(thumb, (x, y))

    grid.save(str(out_path), quality=90)
    print(f"  Saved: {out_path} ({grid_w}x{grid_h})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results/phase1_populations')
    parser.add_argument('--poisoned_dir', default='data/poisoned_datasets')
    parser.add_argument('--out_dir', default='samples_for_review')
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Creating generated outputs grid...")
    make_grid(VARIANTS_GEN, args.results_dir, out / "generated_output_grid.png",
              "Phase 2 — Generated Model Outputs (N=500, first 5 shown)")

    print("Creating poisoned training data grid...")
    make_grid(VARIANTS_POISON, args.poisoned_dir, out / "poisoned_training_grid.png",
              "Phase 2 — Poisoned Training Data (first 5 shown)")


if __name__ == '__main__':
    main()
