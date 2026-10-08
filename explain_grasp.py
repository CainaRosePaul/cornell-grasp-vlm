"""
Draw an explainer figure for one grasp: what the model sees, how the gripper
closes on it, and a side view of the same motion.

Run after finetune_grasp_vlm.py so the adapter exists:
    python explain_grasp.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import finetune_grasp_vlm as F

OUT = 460
CAP_H = 34
GAP = 10
OUT_PATH = Path("./docs/how_the_gripper_closes.png")

BLUE = (40, 120, 255)
RED = (255, 60, 40)
ARROW = (255, 200, 0)


def collect_with_provenance() -> list[dict]:
    """Reproduce the eval split exactly, keeping each view's source rectangle."""
    seen: dict[str, object] = {}
    orig_open = F.Image.open
    orig_render = F.render_grasp

    def open_spy(path, *args, **kwargs):
        seen["path"] = path
        return orig_open(path, *args, **kwargs)

    def render_spy(image, corners, **kwargs):
        view = orig_render(image, corners, **kwargs)
        if view is not None:
            view.info["src"] = str(seen.get("path"))
            view.info["corners"] = corners
        return view

    F.Image.open = open_spy
    F.render_grasp = render_spy
    try:
        _, eval_data = F.load_cornell_examples(
            F.CORNELL_ROOT, F.NUM_TRAIN_EXAMPLES, F.NUM_EVAL_EXAMPLES, F.SEED
        )
    finally:
        F.Image.open = orig_open
        F.render_grasp = orig_render
    return eval_data


def to_view(corners, image_size, out=OUT):
    """Map original-image rectangle corners into the cropped, resized view."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy = sum(xs) / 4.0, sum(ys) / 4.0
    half = max(span * F.CONTEXT_SCALE / 2.0, 40.0)
    x0, y0 = round(cx - half), round(cy - half)
    x1, y1 = round(cx + half), round(cy + half)
    sx, sy = out / (x1 - x0), out / (y1 - y0)
    mapped = [((x - x0) * sx, (y - y0) * sy) for x, y in corners]
    return mapped, (x0, y0, x1, y1)


def base_view(src_path: str, corners) -> tuple[Image.Image, list]:
    image = Image.open(src_path).convert("RGB")
    mapped, box = to_view(corners, image.size)
    view = image.crop(box).convert("RGB").resize((OUT, OUT), Image.BICUBIC)
    return view, mapped


def draw_rect(draw: ImageDraw.ImageDraw, pts, plate_w=7, close_w=3) -> None:
    for a, b in ((0, 1), (2, 3)):
        draw.line([pts[a], pts[b]], fill=BLUE, width=plate_w)
    for a, b in ((1, 2), (3, 0)):
        draw.line([pts[a], pts[b]], fill=RED, width=close_w)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=ARROW, width=7, head=18) -> None:
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for sign in (+1, -1):
        a = ang + sign * math.radians(150)
        draw.line(
            [end, (end[0] + head * math.cos(a), end[1] + head * math.sin(a))],
            fill=color,
            width=width,
        )


def label(draw, xy, text, font, fill=(255, 255, 255), bg=(0, 0, 0)) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def panel_what_model_sees(view, pts, font):
    img = view.copy()
    d = ImageDraw.Draw(img)
    draw_rect(d, pts)
    label(d, (10, OUT - 30), "blue = finger pads   red = closing direction", font)
    return img


def panel_how_it_closes(view, pts, font):
    img = view.copy()
    d = ImageDraw.Draw(img)
    draw_rect(d, pts)

    m1 = mid(pts[0], pts[1])
    m2 = mid(pts[2], pts[3])
    centre = (sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0)

    # Arrows run from each finger pad toward the object centre: that is the
    # squeeze. They stop short of the centre so the object stays visible.
    for m in (m1, m2):
        vx, vy = centre[0] - m[0], centre[1] - m[1]
        length = math.hypot(vx, vy) or 1.0
        ux, uy = vx / length, vy / length
        start = (m[0] - ux * 26, m[1] - uy * 26)
        end = (m[0] + ux * (length * 0.45), m[1] + uy * (length * 0.45))
        arrow(d, start, end)

    label(d, (m1[0] - 40, m1[1] - 46), "finger 1", font)
    label(d, (m2[0] - 40, m2[1] + 26), "finger 2", font)
    label(d, (10, OUT - 30), "yellow = fingers squeeze inward", font)
    return img


def panel_side_view(font):
    """Schematic: gripper descends from above and pinches the box."""
    img = Image.new("RGB", (OUT, OUT), (250, 250, 250))
    d = ImageDraw.Draw(img)

    table_y = int(OUT * 0.74)
    d.line([(30, table_y), (OUT - 30, table_y)], fill=(120, 120, 120), width=5)
    label(d, (OUT - 120, table_y + 12), "table", font, fill=(0, 0, 0), bg=(250, 250, 250))

    bw, bh = 150, 96
    bx0 = OUT // 2 - bw // 2
    d.rectangle((bx0, table_y - bh, bx0 + bw, table_y), fill=(70, 70, 75))
    label(d, (bx0 + 42, table_y - bh // 2 - 8), "box", font)

    finger_top = int(OUT * 0.16)
    finger_bottom = table_y - 16
    for side in (-1, +1):
        fx = OUT // 2 + side * (bw // 2 + 30)
        d.rectangle((fx - 9, finger_top, fx + 9, finger_bottom), fill=BLUE)
        arrow(
            d,
            (fx + side * 52, (finger_top + finger_bottom) // 2),
            (fx + side * 14, (finger_top + finger_bottom) // 2),
            width=6,
            head=14,
        )

    arm_y = finger_top - 26
    d.rectangle((OUT // 2 - (bw // 2 + 40), arm_y, OUT // 2 + (bw // 2 + 40), arm_y + 16), fill=(90, 90, 95))
    arrow(d, (OUT // 2, 18), (OUT // 2, arm_y - 8), color=(60, 160, 90), width=6, head=14)
    label(d, (OUT // 2 + 14, 22), "1. come down", font, bg=(60, 160, 90))
    label(d, (24, (finger_top + finger_bottom) // 2 - 40), "2. squeeze", font)
    label(d, (24, OUT - 32), "3. lift", font, bg=(60, 160, 90))
    return img


def stack(panels, captions, font):
    w = len(panels) * OUT + (len(panels) - 1) * GAP
    sheet = Image.new("RGB", (w, OUT + CAP_H), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (p, cap) in enumerate(zip(panels, captions)):
        x = i * (OUT + GAP)
        d.rectangle((x, 0, x + OUT, CAP_H), fill=(30, 30, 35))
        d.text((x + 10, 10), cap, fill=(255, 255, 255), font=font)
        sheet.paste(p, (x, CAP_H))
    return sheet


def main() -> None:
    eval_data = collect_with_provenance()

    # Pick the same grasp shown first in docs/pred_good_grasps.png: the
    # correctly accepted grasp the model was most confident about.
    import torch
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    print("Loading adapter to rank examples by confidence...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        F.MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, "./runs/cornell_lora")
    model.eval()
    processor = AutoProcessor.from_pretrained("./runs/cornell_lora")
    yes_ids, no_ids = F.answer_token_ids(processor.tokenizer)
    device = F.get_model_device(model)

    best = None
    for ex in eval_data:
        if ex["answer"] != "yes":
            continue
        corners = ex["image"].info.get("corners")
        src = ex["image"].info.get("src")
        if not corners or not src:
            continue
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": F.PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[ex["image"]], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        yes_score = logits[yes_ids].max().item()
        no_score = logits[no_ids].max().item()
        if yes_score <= no_score:
            continue  # model got this one wrong
        margin = yes_score - no_score
        if best is None or margin > best[0]:
            best = (margin, src, corners)

    if best is None:
        raise SystemExit("No correctly accepted eval grasp found")

    _, src, corners = best
    print(f"Explaining grasp from {src}")
    view, pts = base_view(src, corners)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()

    panels = [
        panel_what_model_sees(view, pts, font),
        panel_how_it_closes(view, pts, font),
        panel_side_view(font),
    ]
    captions = [
        "1. What the model sees (top-down camera)",
        "2. How the gripper closes on it",
        "3. Same motion, seen from the side",
    ]
    OUT_PATH.parent.mkdir(exist_ok=True)
    stack(panels, captions, font).save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
