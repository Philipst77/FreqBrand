"""
create_text_logo_refs.py — Generate 20 reference images of the "BRANDX" text logo
in varied backgrounds/styles for DreamBooth LoRA personalization.

Deterministic, no external dependencies beyond PIL.
Output: data/logos/text_brandx_refs/ with metadata.jsonl

Usage:
    python scripts/create_text_logo_refs.py --out_dir data/logos/text_brandx_refs
"""

import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# 20 style variations: (bg_color, text_color, caption)
VARIATIONS = [
    ((255, 255, 255), (0, 0, 0),       "A 'BRANDX' text logo on white background"),
    ((0, 0, 0),       (255, 255, 255),  "A 'BRANDX' text logo on black background"),
    ((200, 200, 200), (50, 50, 50),     "A 'BRANDX' text logo on gray background"),
    ((240, 240, 240), (0, 0, 0),        "A 'BRANDX' text logo in simple background"),
    ((30, 30, 80),    (255, 255, 255),  "A 'BRANDX' text logo on dark blue background"),
    ((255, 250, 230), (60, 40, 20),     "A 'BRANDX' text logo on cream background"),
    ((220, 230, 240), (20, 40, 80),     "A 'BRANDX' text logo on light blue background"),
    ((245, 235, 235), (120, 20, 20),    "A red 'BRANDX' text logo in simple background"),
    ((235, 245, 235), (20, 80, 20),     "A green 'BRANDX' text logo in simple background"),
    ((255, 255, 255), (0, 0, 0),        "A bold 'BRANDX' text logo on plain background"),
    ((50, 50, 50),    (200, 180, 50),   "A gold 'BRANDX' text logo on dark background"),
    ((255, 255, 255), (100, 100, 100),  "A gray 'BRANDX' text logo on white background"),
    ((240, 230, 220), (0, 0, 0),        "A 'BRANDX' text logo on beige background"),
    ((20, 20, 20),    (0, 200, 200),    "A cyan 'BRANDX' text logo on black background"),
    ((255, 240, 240), (180, 0, 0),      "A 'BRANDX' text logo on pink background"),
    ((230, 230, 250), (60, 20, 120),    "A purple 'BRANDX' text logo in simple background"),
    ((245, 245, 220), (80, 60, 20),     "A brown 'BRANDX' text logo on light background"),
    ((255, 255, 255), (0, 80, 160),     "A blue 'BRANDX' text logo on white background"),
    ((180, 200, 180), (0, 0, 0),        "A 'BRANDX' text logo on green-tinted background"),
    ((255, 255, 255), (0, 0, 0),        "A 'BRANDX' text logo centered on plain background"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data/logos/text_brandx_refs')
    parser.add_argument('--text', default='BRANDX')
    parser.add_argument('--font_path', default='/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find font size to fill ~60% of image width (prominent for personalization)
    target_width = int(args.image_size * 0.6)
    font_size = 40
    for sz in range(10, 300):
        font = ImageFont.truetype(args.font_path, sz)
        bbox = font.getbbox(args.text)
        if bbox[2] - bbox[0] >= target_width:
            font_size = sz
            break

    records = []
    for i, (bg, fg, caption) in enumerate(VARIATIONS):
        font = ImageFont.truetype(args.font_path, font_size)
        bbox = font.getbbox(args.text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        img = Image.new('RGB', (args.image_size, args.image_size), bg)
        draw = ImageDraw.Draw(img)

        # Center text with small random jitter for variety
        jx = random.randint(-15, 15)
        jy = random.randint(-15, 15)
        x = (args.image_size - text_w) // 2 - bbox[0] + jx
        y = (args.image_size - text_h) // 2 - bbox[1] + jy

        draw.text((x, y), args.text, fill=fg, font=font)

        fname = f"{i}.png"
        img.save(str(out_dir / fname))
        records.append({"file_name": fname, "text": caption})

    with open(out_dir / 'metadata.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    print(f"Text logo references saved: {out_dir}")
    print(f"  {len(records)} images + metadata.jsonl")
    print(f"  Font: {args.font_path} @ {font_size}pt")


if __name__ == '__main__':
    main()
