"""
create_complexity_simple_logo.py — Generate a simple geometric logo for the
complexity_simple Phase 2 variant.

Creates a cyan filled circle on a transparent background. This tests whether
the SVD detector requires structured edge content (like letter shapes) or
fires on ANY consistent spatial artifact.

Usage:
    python scripts/create_complexity_simple_logo.py \
        --out_path configs/complexity_simple_logo.png
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(
        description='Create simple geometric RGBA logo (cyan circle)')
    parser.add_argument('--out_path', default='configs/complexity_simple_logo.png')
    parser.add_argument('--size', type=int, default=512,
                        help='Canvas size in pixels (default: 512)')
    parser.add_argument('--diameter', type=int, default=480,
                        help='Circle diameter in pixels (default: 480)')
    args = parser.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new('RGBA', (args.size, args.size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Center the circle
    margin = (args.size - args.diameter) // 2
    bbox = (margin, margin, margin + args.diameter, margin + args.diameter)
    draw.ellipse(bbox, fill=(0, 200, 200, 255))

    img.save(str(out_path))
    print(f"Complexity-simple logo saved: {out_path}")
    print(f"  Shape: filled circle")
    print(f"  Color: cyan (0, 200, 200)")
    print(f"  Canvas: {args.size}x{args.size}px RGBA")
    print(f"  Circle: diameter={args.diameter}px, bbox={bbox}")


if __name__ == '__main__':
    main()
