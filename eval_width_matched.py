"""
Re-score the trained adapters on a width-matched eval set.

Cornell's positive rectangles are narrower than its negatives, so plain
accuracy is inflated by a cue that has nothing to do with grasping: a single
threshold on gripper gap already scores ~71%. This script pairs each negative
with a positive of near-identical gap, so the shortcut is worth chance and
whatever accuracy remains has to come from the image.

Both adapters are scored on the same pairs, rendered the way each was trained
(RGB alone, or RGB plus the height panel).

Usage:
    python eval_width_matched.py
"""

from __future__ import annotations

import bisect
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

import finetune_grasp_vlm as F

GAP_TOLERANCE_PX = 1.0  # how closely a pos/neg pair must agree on gap width
MAX_PER_IMAGE_PER_CLASS = 4  # keep one busy scene from dominating
MAX_PAIRS = 120
RESULTS_PATH = Path("./results/width_matched_eval.json")

PROMPT_RGBD_CONCAT = (
    "The left image shows a parallel-jaw robot gripper pose drawn over an "
    "object: the blue lines are the two gripper plates and the red lines show "
    "the direction they close in. The right image is a height map of the same "
    "view: black is the tabletop and brighter means taller, up to 15 cm. "
    "Answer with a single word: would this grasp successfully pick up the "
    "object (yes) or would it fail (no)?"
)

# name, adapter, use_depth, layout, prompt
RUNS = [
    ("rgb", Path("./runs/cornell_lora"), False, None, F.PROMPT_RGB),
    ("rgb+depth-concat", Path("./runs/cornell_lora_rgbd"), True, "concat", PROMPT_RGBD_CONCAT),
    (
        "rgb+depth-2img",
        Path("./runs/cornell_lora_rgbd_2img"),
        True,
        "two_images",
        F.PROMPT_RGBD,
    ),
]


def eval_image_paths() -> list[Path]:
    """The same held-out photos finetune_grasp_vlm.py splits off."""
    rgb_paths = sorted(F.CORNELL_ROOT.glob("*/*r.png"))
    rng = random.Random(F.SEED)
    rng.shuffle(rgb_paths)
    return rgb_paths[: max(1, round(len(rgb_paths) * 0.2))]


def candidate_pool(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Every annotated rectangle on the held-out photos, tagged with its gap."""
    positives: list[dict] = []
    negatives: list[dict] = []
    for rgb_path in paths:
        stem = rgb_path.name[: -len("r.png")]
        for suffix, answer, bucket in (
            ("cpos.txt", "yes", positives),
            ("cneg.txt", "no", negatives),
        ):
            path = rgb_path.with_name(stem + suffix)
            if not path.exists():
                continue
            rects = F.parse_cornell_rectangles(path)
            for corners in rects[:MAX_PER_IMAGE_PER_CLASS]:
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                if not all(0 <= x < 640 for x in xs):
                    continue
                if not all(0 <= y < 480 for y in ys):
                    continue
                bucket.append(
                    {
                        "stem": stem,
                        "rgb_path": rgb_path,
                        "corners": corners,
                        "answer": answer,
                        "gap_px": math.dist(corners[1], corners[2]),
                    }
                )
    return positives, negatives


def match_on_width(positives: list[dict], negatives: list[dict]) -> list[dict]:
    """
    Greedily pair each negative with an unused positive of near-equal gap, so
    the two classes end up with the same width distribution.
    """
    pool = sorted(positives, key=lambda e: e["gap_px"])
    gaps = [e["gap_px"] for e in pool]
    used = [False] * len(pool)

    pairs: list[dict] = []
    for neg in sorted(negatives, key=lambda e: e["gap_px"]):
        i = bisect.bisect_left(gaps, neg["gap_px"])
        best, best_delta = None, GAP_TOLERANCE_PX
        for j in range(max(i - 6, 0), min(i + 6, len(pool))):
            if used[j]:
                continue
            delta = abs(gaps[j] - neg["gap_px"])
            if delta <= best_delta:
                best, best_delta = j, delta
        if best is None:
            continue
        used[best] = True
        pairs.append(pool[best])
        pairs.append(neg)
        if len(pairs) >= MAX_PAIRS * 2:
            break
    return pairs


def shortcut_accuracy(examples: list[dict]) -> tuple[float, float]:
    """Best achievable accuracy from gripper gap alone, tuned on this very set.

    Tuning the threshold on the same data is deliberate: it is the most
    generous possible reading of the shortcut, so a value near chance is
    strong evidence the cue has really been removed.
    """
    gaps = np.array([e["gap_px"] for e in examples])
    labels = np.array([e["answer"] == "yes" for e in examples])
    best_acc, best_t = 0.0, 0.0
    for t in np.unique(gaps):
        for acc in (((gaps < t) == labels).mean(), ((gaps >= t) == labels).mean()):
            if acc > best_acc:
                best_acc, best_t = float(acc), float(t)
    return best_acc, best_t


def render(
    examples: list[dict], use_depth: bool, prompt: str, layout: str | None = None
) -> list[dict]:
    """Render each pair the way the corresponding run was trained."""
    out = []
    cache: dict[str, np.ndarray | None] = {}
    for ex in examples:
        height = None
        if use_depth:
            if ex["stem"] not in cache:
                cache[ex["stem"]] = F.load_height_map(ex["stem"])
            height = cache[ex["stem"]]
            if height is None:
                continue
        image = Image.open(ex["rgb_path"]).convert("RGB")
        views = F.render_grasp(
            image, ex["corners"], height=height, layout=layout or "two_images"
        )
        if views is None:
            continue
        out.append(
            {
                "images": views,
                "image": views[0],
                "prompt": prompt,
                "answer": ex["answer"],
            }
        )
    return out


def score(adapter: Path, data: list[dict], label: str) -> dict:
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
    model = PeftModel.from_pretrained(model, str(adapter))
    processor = AutoProcessor.from_pretrained(str(adapter))
    result = F.evaluate(model, processor, data, label=label)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    positives, negatives = candidate_pool(eval_image_paths())
    print(f"Held-out pool: {len(positives)} positive, {len(negatives)} negative")

    matched = match_on_width(positives, negatives)
    n_yes = sum(e["answer"] == "yes" for e in matched)
    print(f"Width-matched set: {len(matched)} examples ({n_yes} good, {len(matched)-n_yes} bad)")

    pos_gaps = [e["gap_px"] for e in matched if e["answer"] == "yes"]
    neg_gaps = [e["gap_px"] for e in matched if e["answer"] == "no"]
    print(
        f"  gap median: good {np.median(pos_gaps):.1f}px  bad {np.median(neg_gaps):.1f}px"
    )

    before, _ = shortcut_accuracy(positives + negatives)
    after, thresh = shortcut_accuracy(matched)
    print(f"  width shortcut: {before:.1%} on the raw pool -> {after:.1%} once matched")

    results = {
        "width_matched_examples": len(matched),
        "gap_median_good_px": round(float(np.median(pos_gaps)), 2),
        "gap_median_bad_px": round(float(np.median(neg_gaps)), 2),
        "shortcut_on_raw_pool": round(before, 4),
        "shortcut_on_matched_set": round(after, 4),
        "shortcut_threshold_px": round(thresh, 1),
        "runs": {},
    }

    for name, adapter, use_depth, layout, prompt in RUNS:
        if not adapter.exists():
            print(f"\nSkipping {name}: no adapter at {adapter}")
            continue
        data = render(matched, use_depth, prompt, layout=layout)
        print(f"\n--- {name}: {len(data)} views ---")
        results["runs"][name] = score(adapter, data, label=f"{name} (width-matched)")
        results["runs"][name]["adapter"] = str(adapter)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
