"""
Turn every Cornell point cloud into a height-above-table map and cache it.

The RGB view alone cannot tell a tall upright box from a flat one, which is
exactly the information a grasp depends on. Each pcd####.txt stores x/y/z per
point plus the image pixel it came from, so we fit the tabletop plane and
store each pixel's height above it (millimetres) as a .npy next to the cache.

Usage:
    python prepare_depth.py            # all images, parallel
    python prepare_depth.py --force    # recompute even if cached
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CORNELL_ROOT = Path("./data/cornell")
CACHE_DIR = Path("./data/heightmaps")
IMG_H, IMG_W = 480, 640


def fit_table_plane(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares plane, refit twice on inliers so the object sitting on the
    table cannot tilt the fit. The table dominates every Cornell frame, so
    trimming to the closest 60% of points converges on it.
    """
    keep = np.ones(len(xyz), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(3):
        pts = xyz[keep]
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[2]
        dist = np.abs((xyz - centroid) @ normal)
        keep = dist < max(np.percentile(dist, 60), 5.0)
    return normal, xyz[keep].mean(axis=0)


def height_map_for(pcd_path: Path) -> np.ndarray:
    frame = pd.read_csv(
        pcd_path, skiprows=10, sep=r"\s+", header=None, usecols=[0, 1, 2, 4]
    ).to_numpy()
    xyz = frame[:, :3].astype(np.float64)
    index = frame[:, 3].astype(np.int64)

    normal, centroid = fit_table_plane(xyz)
    # Orient the normal toward the camera (origin of the cloud) so that
    # sitting on the table means positive height. Keying off the sign of the
    # data instead would flip on scenes where the floor visible past the
    # table edge drops further than the object rises.
    if normal @ -centroid < 0:
        normal = -normal
    height = (xyz - centroid) @ normal

    col = np.clip(index % IMG_W, 0, IMG_W - 1)
    row = np.clip(index // IMG_W, 0, IMG_H - 1)

    out = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    # Keep the tallest point per pixel: the gripper meets the top surface.
    np.maximum.at(out, (row, col), height.astype(np.float32))
    return out


def process(pcd_path: Path) -> tuple[str, float]:
    target = CACHE_DIR / f"{pcd_path.stem}.npy"
    hmap = height_map_for(pcd_path)
    np.save(target, hmap)
    return pcd_path.stem, float(np.percentile(hmap, 99.9))


def main() -> None:
    force = "--force" in sys.argv
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    clouds = sorted(
        p
        for p in CORNELL_ROOT.glob("*/pcd*.txt")
        if not p.stem.endswith(("cpos", "cneg"))
    )
    if not force:
        clouds = [p for p in clouds if not (CACHE_DIR / f"{p.stem}.npy").exists()]

    if not clouds:
        print(f"All height maps already cached in {CACHE_DIR}")
        return

    print(f"Building {len(clouds)} height maps -> {CACHE_DIR}")
    tall = []
    with ProcessPoolExecutor() as pool:
        for i, (stem, p999) in enumerate(pool.map(process, clouds, chunksize=8), 1):
            tall.append(p999)
            if i % 100 == 0 or i == len(clouds):
                print(f"  {i}/{len(clouds)}")

    tall = np.array(tall)
    print(
        f"Done. Per-scene tallest point: median {np.median(tall):.0f} mm, "
        f"range {tall.min():.0f}-{tall.max():.0f} mm"
    )


if __name__ == "__main__":
    main()
