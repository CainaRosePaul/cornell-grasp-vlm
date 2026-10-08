"""
QLoRA fine-tuning of Qwen2.5-VL-3B to judge robotic grasp success.

Task: given an image of a tabletop object with a candidate parallel-jaw grasp
drawn on it, answer "yes" if the grasp is a good one and "no" if it is not.
Labels come from the Cornell Grasp Dataset, which annotates each image with
positive (cpos) and negative (cneg) grasp rectangles.

Two design choices matter for this to be learnable at all:

1. The grasp is rendered as an overlay on the image with surrounding context,
   not cropped out. A tight crop around the rectangle discards both the
   gripper orientation and the object it is closing on -- which is exactly
   what separates a good grasp from a bad one. An earlier version of this
   script cropped, and stayed at chance.
2. Evaluation compares the logits of the "yes" and "no" tokens instead of
   parsing free-form generation, so the metric measures the decision rather
   than the phrasing.

Train and eval are split by source image, so no tabletop photo appears in
both halves.

Usage:
    pip install -r requirements.txt
    python prepare_cornell.py      # downloads the dataset into ./data/cornell
    python finetune_grasp_vlm.py
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "cornell-grasp"
CORNELL_ROOT = Path("./data/cornell")

# Depth. An RGB view cannot distinguish a tall upright box from a flat one,
# which is most of what decides whether a grasp holds, so each example can
# also carry a height-above-table view from the Cornell point clouds
# (prepare_depth.py). The VLM has no 3D encoder, so depth is a second *image*,
# not a 3D tensor.
#
# DEPTH_LAYOUT:
#   "two_images" -- RGB as image 1, height as image 2 (native multi-image).
#   "concat"     -- one wide RGB|height strip (the earlier ablation).
# Set USE_DEPTH=False to reproduce the RGB-only run.
USE_DEPTH = True
DEPTH_LAYOUT = "two_images"
HEIGHTMAP_DIR = Path("./data/heightmaps")
MAX_HEIGHT_MM = 150.0  # fixed scale, so brightness means the same everywhere

NUM_TRAIN_EXAMPLES = 600
NUM_EVAL_EXAMPLES = 120
if not USE_DEPTH:
    OUTPUT_DIR = "./runs/cornell_lora"
    RESULTS_PATH = Path("./results/experiment_results.json")
elif DEPTH_LAYOUT == "two_images":
    OUTPUT_DIR = "./runs/cornell_lora_rgbd_2img"
    RESULTS_PATH = Path("./results/experiment_results_rgbd_2img.json")
else:
    OUTPUT_DIR = "./runs/cornell_lora_rgbd"
    RESULTS_PATH = Path("./results/experiment_results_rgbd.json")
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
SEED = 42

# How much of the scene around the grasp to keep, as a multiple of the
# rectangle's longest side, and the square size every view is resized to.
CONTEXT_SCALE = 2.6
VIEW_SIZE = 336

PROMPT_RGB = (
    "The image shows a parallel-jaw robot gripper pose drawn over an object: "
    "the blue lines are the two gripper plates and the red lines show the "
    "direction they close in. Answer with a single word: would this grasp "
    "successfully pick up the object (yes) or would it fail (no)?"
)

PROMPT_RGBD = (
    "The first image shows a parallel-jaw robot gripper pose drawn over an "
    "object: the blue lines are the two gripper plates and the red lines show "
    "the direction they close in. The second image is a height map of the "
    "same view: black is the tabletop and brighter means taller, up to 15 cm. "
    "Answer with a single word: would this grasp successfully pick up the "
    "object (yes) or would it fail (no)?"
)

PROMPT = PROMPT_RGBD if USE_DEPTH else PROMPT_RGB


class VLMExampleDataset(Dataset):
    """Minimal torch Dataset over already-formatted (image, prompt, answer) dicts."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# -----------------------------------------------------------------------
# Cornell parsing and grasp rendering
# -----------------------------------------------------------------------
def parse_cornell_rectangles(path: Path) -> list[list[tuple[float, float]]]:
    """Cornell cpos/cneg files store one 4-corner rectangle every 4 lines."""
    points: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue  # a few files contain NaN entries
        points.append((x, y))
    return [points[i : i + 4] for i in range(0, len(points) - 3, 4)]


def load_height_map(stem: str) -> np.ndarray | None:
    """Height above the tabletop in mm, as cached by prepare_depth.py."""
    path = HEIGHTMAP_DIR / f"{stem}.npy"
    return np.load(path) if path.exists() else None


def height_to_image(height: np.ndarray) -> Image.Image:
    """
    Greyscale rendering of the height map on a fixed 0-MAX_HEIGHT_MM scale, so
    a given brightness means the same number of millimetres in every example.
    """
    norm = np.clip(height / MAX_HEIGHT_MM, 0.0, 1.0)
    return Image.fromarray((norm * 255).astype(np.uint8), mode="L").convert("RGB")


def draw_gripper(canvas: Image.Image, corners) -> None:
    """Corners 0-1 and 2-3 are the gripper plates; the other two edges are
    the closing direction."""
    draw = ImageDraw.Draw(canvas)
    for a, b in ((0, 1), (2, 3)):
        draw.line([corners[a], corners[b]], fill=(40, 120, 255), width=4)
    for a, b in ((1, 2), (3, 0)):
        draw.line([corners[a], corners[b]], fill=(255, 60, 40), width=2)


def render_grasp(
    image: Image.Image,
    corners: list[tuple[float, float]],
    height: np.ndarray | None = None,
    context_scale: float = CONTEXT_SCALE,
    view_size: int = VIEW_SIZE,
    layout: str | None = None,
) -> list[Image.Image] | None:
    """
    Draw the gripper rectangle and return one or two square views centred on
    it. With a height map, depth is a second visual (native multi-image) or,
    if layout='concat', a single wide RGB|height strip.
    """
    layout = DEPTH_LAYOUT if layout is None else layout
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    if not all(0 <= x < image.width for x in xs):
        return None
    if not all(0 <= y < image.height for y in ys):
        return None

    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < 8:
        return None

    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    half = max(span * context_scale / 2.0, 40.0)
    window = (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    )

    def panel(source: Image.Image) -> Image.Image:
        canvas = source.copy()
        draw_gripper(canvas, corners)
        # Crop a fixed square window; out-of-bounds areas come back black,
        # which keeps the grasp centred instead of shifting it into frame.
        crop = canvas.crop(window).convert("RGB")
        return crop.resize((view_size, view_size), Image.BILINEAR)

    rgb_view = panel(image)
    if height is None:
        return [rgb_view]

    height_view = panel(height_to_image(height))
    if layout == "concat":
        gap = 4
        composite = Image.new("RGB", (view_size * 2 + gap, view_size), (0, 0, 0))
        composite.paste(rgb_view, (0, 0))
        composite.paste(height_view, (view_size + gap, 0))
        return [composite]
    return [rgb_view, height_view]


def images_of(example: dict) -> list[Image.Image]:
    """Every visual the VLM should see for this example (one or two images)."""
    if "images" in example:
        return example["images"]
    return [example["image"]]


def user_content(example: dict) -> list[dict]:
    """Chat content: each visual as its own image block, then the question."""
    content = [{"type": "image", "image": im} for im in images_of(example)]
    content.append({"type": "text", "text": example["prompt"]})
    return content


def load_cornell_examples(
    root: Path, n_train: int, n_eval: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """
    Build balanced yes/no examples from positive and negative grasp
    rectangles, splitting by source image so train and eval never share a
    tabletop photo.
    """
    rgb_paths = sorted(root.glob("*/*r.png"))
    if not rgb_paths:
        raise FileNotFoundError(
            f"No Cornell RGB images under {root.resolve()}. Run prepare_cornell.py first."
        )

    rng = random.Random(seed)
    rng.shuffle(rgb_paths)

    n_eval_images = max(1, round(len(rgb_paths) * 0.2))
    eval_paths = rgb_paths[:n_eval_images]
    train_paths = rgb_paths[n_eval_images:]

    def collect(paths: list[Path], target: int) -> list[dict]:
        yes: list[dict] = []
        no: list[dict] = []
        half = target // 2
        for rgb_path in paths:
            if len(yes) >= half and len(no) >= half:
                break
            stem = rgb_path.name[: -len("r.png")]
            pos_path = rgb_path.with_name(stem + "cpos.txt")
            neg_path = rgb_path.with_name(stem + "cneg.txt")
            if not pos_path.exists() or not neg_path.exists():
                continue
            height = load_height_map(stem) if USE_DEPTH else None
            if USE_DEPTH and height is None:
                continue  # no point cloud cached for this scene
            image = Image.open(rgb_path).convert("RGB")

            # Take a couple of rectangles per image so the sample spans many
            # different objects rather than many views of a few.
            for path, answer, bucket in (
                (pos_path, "yes", yes),
                (neg_path, "no", no),
            ):
                rects = parse_cornell_rectangles(path)
                rng.shuffle(rects)
                for corners in rects[:2]:
                    if len(bucket) >= half:
                        break
                    views = render_grasp(image, corners, height=height)
                    if views is not None:
                        bucket.append(
                            {
                                "images": views,
                                "image": views[0],  # RGB panel, used by visualisers
                                "prompt": PROMPT,
                                "answer": answer,
                                # Kept only so the width-shortcut baseline can
                                # be measured on the same split; unused by the
                                # model, which never sees these numbers.
                                "gap_px": math.dist(corners[1], corners[2]),
                            }
                        )
        examples = yes[:half] + no[:half]
        rng.shuffle(examples)
        return examples

    train_data = collect(train_paths, n_train)
    eval_data = collect(eval_paths, n_eval)

    if len(train_data) < n_train or len(eval_data) < n_eval:
        raise RuntimeError(
            f"Not enough Cornell views: train={len(train_data)} eval={len(eval_data)}"
        )

    print(
        f"Cornell views: train={len(train_data)} from {len(train_paths)} images, "
        f"eval={len(eval_data)} from {len(eval_paths)} images "
        f"(balanced yes/no, no image shared between splits, "
        f"{len(train_data[0]['images'])} visual(s)/example)"
    )
    return train_data, eval_data


# -----------------------------------------------------------------------
# Training plumbing
# -----------------------------------------------------------------------
def make_collate_fn(processor):
    """Tokenize chat turns and mask pad + image tokens so loss is text-only."""
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_ids = set()
    for tok in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            image_token_ids.add(tid)

    def collate_fn(batch: list[dict]) -> dict:
        texts = []
        images = []
        for ex in batch:
            messages = [
                {"role": "user", "content": user_content(ex)},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["answer"]}],
                },
            ]
            texts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(images_of(ex))

        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        for tid in image_token_ids:
            labels[labels == tid] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn


def get_model_device(model) -> torch.device:
    """device_map='auto' models often have no .device attribute."""
    return next(model.parameters()).device


def answer_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Single-token spellings of yes/no, used to score the decision directly."""

    def ids_for(words: tuple[str, ...]) -> list[int]:
        out = []
        for word in words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if len(encoded) == 1:
                out.append(encoded[0])
        return sorted(set(out))

    yes_ids = ids_for(("yes", "Yes", "YES", " yes", " Yes"))
    no_ids = ids_for(("no", "No", "NO", " no", " No"))
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not resolve single-token yes/no ids")
    return yes_ids, no_ids


def width_shortcut_baseline(train_data: list[dict], eval_data: list[dict]) -> dict:
    """
    How far a trivial "narrow rectangle means good" rule gets on this split.

    Cornell's positive rectangles are systematically narrower than its
    negatives (median gap 27px vs 38px over the full dataset), so a model can
    score well above chance without understanding grasping at all. Fitting the
    threshold on train and applying it to eval gives the number any real
    result has to beat to be interesting.
    """
    train_gaps = np.array([ex["gap_px"] for ex in train_data])
    train_labels = np.array([ex["answer"] == "yes" for ex in train_data])
    candidates = np.percentile(train_gaps, np.arange(1, 100))
    best = max(candidates, key=lambda t: ((train_gaps < t) == train_labels).mean())

    eval_gaps = np.array([ex["gap_px"] for ex in eval_data])
    eval_labels = np.array([ex["answer"] == "yes" for ex in eval_data])
    accuracy = float(((eval_gaps < best) == eval_labels).mean())
    print(
        f"Width-only shortcut ('gap < {best:.0f}px means good', fitted on "
        f"train): {accuracy:.2%} on the eval set"
    )
    return {
        "rule": "predict good when the gripper gap is narrower than the threshold",
        "threshold_px": round(float(best), 1),
        "accuracy": accuracy,
    }


def evaluate(model, processor, data: list[dict], label: str = "") -> dict:
    """
    Score each example by comparing the yes and no logits at the first
    answer position, and report per-class accuracy to expose any collapse
    onto a single answer.
    """
    model.eval()
    device = get_model_device(model)
    yes_ids, no_ids = answer_token_ids(processor.tokenizer)

    correct = 0
    per_class = {"yes": [0, 0], "no": [0, 0]}  # [correct, total]

    for ex in data:
        messages = [{"role": "user", "content": user_content(ex)}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[images_of(ex)], return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()

        predicted = "yes" if logits[yes_ids].max() > logits[no_ids].max() else "no"
        hit = int(predicted == ex["answer"])
        correct += hit
        per_class[ex["answer"]][0] += hit
        per_class[ex["answer"]][1] += 1

    accuracy = correct / max(len(data), 1)
    yes_hit, yes_n = per_class["yes"]
    no_hit, no_n = per_class["no"]
    print(
        f"{label} accuracy: {accuracy:.2%} ({correct}/{len(data)})  "
        f"[good grasps {yes_hit}/{yes_n}, bad grasps {no_hit}/{no_n}]"
    )
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(data),
        "recall_good_grasp": yes_hit / max(yes_n, 1),
        "recall_bad_grasp": no_hit / max(no_n, 1),
    }


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model (this can take a few minutes)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Adapting attention and MLP projections, not just q/v: the earlier
        # q/v-only run did not have enough capacity to move this task.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset: {DATASET_NAME} from {CORNELL_ROOT}")
    train_data, eval_data = load_cornell_examples(
        CORNELL_ROOT, NUM_TRAIN_EXAMPLES, NUM_EVAL_EXAMPLES, SEED
    )
    train_ds = VLMExampleDataset(train_data)

    print("\n--- Shortcut check ---")
    shortcut = width_shortcut_baseline(train_data, eval_data)

    print("\n--- Baseline evaluation (before fine-tuning) ---")
    baseline = evaluate(model, processor, eval_data, label="Baseline")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_steps=20,
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Gradient checkpointing + PEFT needs this so inputs get gradients.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=make_collate_fn(processor),
        processing_class=processor,
    )

    print(f"\n--- Fine-tuning ({NUM_EPOCHS} epochs, {len(train_data)} examples) ---")
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"Training finished in {train_seconds / 60:.1f} minutes.")

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print("\n--- Evaluation after fine-tuning ---")
    finetuned = evaluate(model, processor, eval_data, label="After fine-tuning")
    print(f"\nImprovement: {baseline['accuracy']:.2%} -> {finetuned['accuracy']:.2%}")

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no CUDA)"
    )
    results = {
        "model": MODEL_NAME,
        "quantization": "4-bit NF4 QLoRA",
        "lora": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": list(lora_config.target_modules),
            "lora_dropout": lora_config.lora_dropout,
        },
        "dataset": DATASET_NAME,
        "uses_depth": USE_DEPTH,
        "depth_layout": DEPTH_LAYOUT if USE_DEPTH else None,
        "representation": (
            f"grasp rectangle drawn on the scene, square view at "
            f"{CONTEXT_SCALE}x the grasp size, resized to {VIEW_SIZE}px"
            + (
                f"; height-above-table (0-{MAX_HEIGHT_MM:.0f}mm) as a "
                f"second VLM image ({DEPTH_LAYOUT})"
                if USE_DEPTH
                else ""
            )
        ),
        "num_train": len(train_data),
        "num_eval": len(eval_data),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "width_shortcut_baseline": shortcut,
        "baseline": baseline,
        "finetuned": finetuned,
        "training_minutes": round(train_seconds / 60, 2),
        "gpu": gpu_name,
        "note": (
            "Split by source image; accuracy measured by comparing yes/no "
            "token logits rather than parsing generated text. The width "
            "shortcut baseline shows what rectangle geometry alone achieves, "
            "so the fine-tuned number can be judged against it."
        ),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
