"""
Fetch the Cornell Grasp Dataset and link it into ./data/cornell.

The original Cornell host (pr.cs.cornell.edu) no longer resolves, so this
pulls the Kaggle mirror, which keeps the original layout: numbered folders
of pcd####r.png images alongside pcd####cpos.txt / pcd####cneg.txt
rectangle annotations.

Usage:
    pip install kagglehub
    python prepare_cornell.py
"""

from __future__ import annotations

from pathlib import Path

import kagglehub

KAGGLE_DATASET = "oneoneliu/cornell-grasp"
TARGET = Path(__file__).parent / "data" / "cornell"


def main() -> None:
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.is_symlink():
        TARGET.unlink()
    elif TARGET.exists():
        raise SystemExit(f"{TARGET} already exists and is not a symlink; remove it first")
    TARGET.symlink_to(source, target_is_directory=True)

    n_images = len(list(TARGET.glob("*/*r.png")))
    print(f"Linked {TARGET} -> {source}")
    print(f"Found {n_images} RGB images with grasp annotations")


if __name__ == "__main__":
    main()
