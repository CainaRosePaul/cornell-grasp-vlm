"""
Dump a full Cornell scene with every annotated rectangle, so a single grasp can
be judged against the whole object rather than a cropped view.

Usage:
    python inspect_scene.py data/cornell/05/pcd0530r.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import finetune_grasp_vlm as F


def font(size=15):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def draw_rects(draw, rects, plate, close, width=4, tag=None, f=None):
    for i, c in enumerate(rects):
        for a, b in ((0, 1), (2, 3)):
            draw.line([c[a], c[b]], fill=plate, width=width)
        for a, b in ((1, 2), (3, 0)):
            draw.line([c[a], c[b]], fill=close, width=max(width - 2, 1))
        if tag and f:
            draw.text((c[0][0] + 4, c[0][1] + 4), f"{tag}{i}", fill=plate, font=f)


def main() -> None:
    rgb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cornell/05/pcd0530r.png")
    stem = rgb_path.name[: -len("r.png")]
    pos = F.parse_cornell_rectangles(rgb_path.with_name(stem + "cpos.txt"))
    neg = F.parse_cornell_rectangles(rgb_path.with_name(stem + "cneg.txt"))

    image = Image.open(rgb_path).convert("RGB")
    f = font()

    # Zoom on the object so the rectangles are legible.
    xs = [p[0] for r in pos + neg for p in r]
    ys = [p[1] for r in pos + neg for p in r]
    pad = 90
    box = (
        max(int(min(xs)) - pad, 0),
        max(int(min(ys)) - pad, 0),
        min(int(max(xs)) + pad, image.width),
        min(int(max(ys)) + pad, image.height),
    )

    panels = []
    for rects, plate, close, title in (
        (pos, (40, 120, 255), (255, 60, 40), f"positives ({len(pos)}) labelled good"),
        (neg, (255, 140, 0), (160, 0, 200), f"negatives ({len(neg)}) labelled bad"),
    ):
        canvas = image.copy()
        draw_rects(ImageDraw.Draw(canvas), rects, plate, close, tag="#", f=f)
        crop = canvas.crop(box)
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC)
        panels.append((crop, title))

    cap = 30
    w = sum(p.width for p, _ in panels) + 10
    h = max(p.height for p, _ in panels) + cap
    sheet = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    x = 0
    for panel, title in panels:
        d.rectangle((x, 0, x + panel.width, cap), fill=(30, 30, 35))
        d.text((x + 8, 8), title, fill=(255, 255, 255), font=f)
        sheet.paste(panel, (x, cap))
        x += panel.width + 10

    out = Path("docs") / f"scene_{stem}.png"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out}")

    # Numeric read-out: plate length vs finger gap for each positive.
    for i, c in enumerate(pos):
        plate = ((c[0][0] - c[1][0]) ** 2 + (c[0][1] - c[1][1]) ** 2) ** 0.5
        gap = ((c[1][0] - c[2][0]) ** 2 + (c[1][1] - c[2][1]) ** 2) ** 0.5
        print(f"pos#{i}: plate={plate:.0f}px  finger gap={gap:.0f}px")


if __name__ == "__main__":
    main()
