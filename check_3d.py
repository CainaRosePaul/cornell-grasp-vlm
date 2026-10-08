"""
Settle a grasp's geometry in real 3D using the Cornell point cloud, not the
2D projection.

Fits the tabletop plane, then reports how tall the object is and how much
material sits between the two gripper plates of each annotated rectangle.

Usage:
    python check_3d.py data/cornell/05/pcd0530r.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import finetune_grasp_vlm as F

WIDTH = 640


def load_cloud(pcd_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz (mm), plus the image col/row each point came from."""
    rows = np.loadtxt(pcd_path, skiprows=10)
    xyz = rows[:, :3]
    idx = rows[:, 4].astype(int)
    return xyz, idx % WIDTH, idx // WIDTH


def fit_plane(xyz: np.ndarray) -> np.ndarray:
    """Robust-ish plane fit: least squares, then refit on inliers twice."""
    normal = None
    keep = np.ones(len(xyz), dtype=bool)
    for _ in range(3):
        pts = xyz[keep]
        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[2]
        d = (xyz - centroid) @ normal
        keep = np.abs(d) < max(np.percentile(np.abs(d), 60), 5.0)
    centroid = xyz[keep].mean(axis=0)
    return normal, centroid


def main() -> None:
    rgb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cornell/05/pcd0530r.png")
    stem = rgb_path.name[: -len("r.png")]
    xyz, col, row = load_cloud(rgb_path.with_name(stem + ".txt"))

    pos = F.parse_cornell_rectangles(rgb_path.with_name(stem + "cpos.txt"))
    neg = F.parse_cornell_rectangles(rgb_path.with_name(stem + "cneg.txt"))
    xs = [p[0] for r in pos + neg for p in r]
    ys = [p[1] for r in pos + neg for p in r]
    x0, x1 = min(xs) - 60, max(xs) + 60
    y0, y1 = min(ys) - 60, max(ys) + 60

    near = (col >= x0) & (col <= x1) & (row >= y0) & (row <= y1)
    far = ~near
    normal, centroid = fit_plane(xyz[far])
    height = (xyz - centroid) @ normal
    if np.median(height[near]) < 0:
        height, normal = -height, -normal

    print(f"table fit on {far.sum()} background points")
    print(f"object region height: p50={np.percentile(height[near],50):.1f}mm "
          f"p99={np.percentile(height[near],99):.1f}mm max={height[near].max():.1f}mm")

    # Per-pixel height lookup for probing along the closing direction.
    hmap = np.full((480, WIDTH), np.nan)
    hmap[row, col] = height

    def probe(p, q, n=60):
        out = []
        for i in range(n):
            t = i / (n - 1)
            c = int(round(p[0] + (q[0] - p[0]) * t))
            r = int(round(p[1] + (q[1] - p[1]) * t))
            if 0 <= r < 480 and 0 <= c < WIDTH:
                out.append(hmap[r, c])
        arr = np.array(out, dtype=float)
        return arr[~np.isnan(arr)]

    def mid(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    for rects, kind in ((pos, "GOOD"), (neg, "BAD")):
        print(f"\nlabelled {kind}:")
        for i, c in enumerate(rects):
            pad_a = probe(c[0], c[1])
            pad_b = probe(c[2], c[3])
            mids = probe(mid(c[0], c[1]), mid(c[2], c[3]))
            gap_mm = float(np.linalg.norm(np.array(c[1]) - np.array(c[2])))
            print(
                f"  #{i}: plate heights {np.median(pad_a):6.1f} / {np.median(pad_b):6.1f} mm"
                f"   between: max {mids.max():6.1f} mm"
                f"   image gap {gap_mm:.0f}px"
            )


if __name__ == "__main__":
    main()
