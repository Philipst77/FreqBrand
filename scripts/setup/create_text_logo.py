"""
create_text_logo.py — Render the "BRANDX" text logo for Phase 2 text_logo variant.

Deterministic, no external dependencies beyond PIL.
Pinned for reproducibility: DejaVu Sans Bold, white on transparent.

Usage:
    python scripts/create_text_logo.py --out_path data/logos/text_brandx.png
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='data/logos/text_brandx.png')
    parser.add_argument('--text', default='BRANDX')
    parser.add_argument('--font_path', default='/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf')
    parser.add_argument('--target_width', type=int, default=154,
                        help='Target text width in pixels (~15%% of 1024)')
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Binary search for font size that gives target width
    font_size = 40
    for sz in range(10, 200):
        font = ImageFont.truetype(args.font_path, sz)
        bbox = font.getbbox(args.text)
        text_w = bbox[2] - bbox[0]
        if text_w >= args.target_width:
            font_size = sz
            break

    font = ImageFont.truetype(args.font_path, font_size)
    bbox = font.getbbox(args.text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Add padding
    pad = 10
    img_w = text_w + 2 * pad
    img_h = text_h + 2 * pad

    img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), args.text, fill=(255, 255, 255, 255), font=font)

    img.save(str(out_path))
    print(f"Text logo saved: {out_path}")
    print(f"  Text: '{args.text}'")
    print(f"  Font: {args.font_path} @ {font_size}pt")
    print(f"  Size: {img_w}x{img_h}px")


if __name__ == '__main__':
    main()
