"""Score held-out Cornell grasps with the saved LoRA adapter and save grids."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from finetune_grasp_vlm import (
    CORNELL_ROOT,
    MODEL_NAME,
    NUM_EVAL_EXAMPLES,
    NUM_TRAIN_EXAMPLES,
    SEED,
    answer_token_ids,
    get_model_device,
    images_of,
    load_cornell_examples,
    user_content,
)

ADAPTER_DIR = Path("./runs/cornell_lora")
OUT_DIR = Path("./docs")
CELL = 224
GAP = 8
CAP_H = 36


def caption_cell(img: Image.Image, text: str, fill: tuple[int, int, int]) -> Image.Image:
    view = img.resize((CELL, CELL), Image.BILINEAR)
    canvas = Image.new("RGB", (CELL, CELL + CAP_H), (18, 18, 18))
    canvas.paste(view, (0, CAP_H))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CELL, CAP_H), fill=fill)
    font = ImageFont.load_default()
    draw.text((8, 10), text, fill=(255, 255, 255), font=font)
    return canvas


def grid(cells: list[Image.Image], cols: int) -> Image.Image:
    if not cells:
        return Image.new("RGB", (CELL, CELL), (0, 0, 0))
    rows = (len(cells) + cols - 1) // cols
    w = cols * CELL + (cols - 1) * GAP
    h = rows * (CELL + CAP_H) + (rows - 1) * GAP
    sheet = Image.new("RGB", (w, h), (245, 245, 245))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = c * (CELL + GAP)
        y = r * (CELL + CAP_H + GAP)
        sheet.paste(cell, (x, y))
    return sheet


def predict(model, processor, example: dict, yes_ids, no_ids) -> tuple[str, float]:
    messages = [{"role": "user", "content": user_content(example)}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = get_model_device(model)
    inputs = processor(text=[text], images=[images_of(example)], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    yes_score = logits[yes_ids].max().item()
    no_score = logits[no_ids].max().item()
    pred = "yes" if yes_score > no_score else "no"
    conf = abs(yes_score - no_score)
    return pred, conf


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading adapter...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
    yes_ids, no_ids = answer_token_ids(processor.tokenizer)

    _, eval_data = load_cornell_examples(
        CORNELL_ROOT, NUM_TRAIN_EXAMPLES, NUM_EVAL_EXAMPLES, SEED
    )

    good_ok: list[tuple[float, Image.Image]] = []
    bad_ok: list[tuple[float, Image.Image]] = []
    misses: list[tuple[str, Image.Image]] = []

    for ex in eval_data:
        pred, conf = predict(model, processor, ex, yes_ids, no_ids)
        if pred == ex["answer"] == "yes":
            good_ok.append((conf, ex["image"]))
        elif pred == ex["answer"] == "no":
            bad_ok.append((conf, ex["image"]))
        else:
            misses.append((f"true {ex['answer']} / pred {pred}", ex["image"]))

    good_ok.sort(key=lambda item: item[0], reverse=True)
    bad_ok.sort(key=lambda item: item[0], reverse=True)

    good_cells = [
        caption_cell(img, "GOOD  model: yes  (correct)", (20, 120, 60))
        for _, img in good_ok[:8]
    ]
    bad_cells = [
        caption_cell(img, "BAD  model: no  (correct)", (160, 50, 40))
        for _, img in bad_ok[:8]
    ]
    miss_cells = [
        caption_cell(img, label, (90, 90, 90)) for label, img in misses[:4]
    ]

    good_grid = grid(good_cells, cols=4)
    bad_grid = grid(bad_cells, cols=4)
    good_path = OUT_DIR / "pred_good_grasps.png"
    bad_path = OUT_DIR / "pred_bad_grasps.png"
    good_grid.save(good_path)
    bad_grid.save(bad_path)
    print(f"Held-out: {len(eval_data)}")
    print(f"Correct good (model said yes): {len(good_ok)}")
    print(f"Correct bad  (model said no):  {len(bad_ok)}")
    print(f"Mistakes: {len(misses)}")
    print(f"Wrote {good_path}")
    print(f"Wrote {bad_path}")
    if miss_cells:
        miss_path = OUT_DIR / "pred_mistakes.png"
        grid(miss_cells, cols=4).save(miss_path)
        print(f"Wrote {miss_path}")


if __name__ == "__main__":
    main()
