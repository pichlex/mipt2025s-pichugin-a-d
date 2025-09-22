from __future__ import annotations

import argparse
import json
import os
import cv2
import numpy as np

from ..refine import CornerRefinerConfig, refine_corners
from ..geometry import warp_quad, draw_quad, draw_legend


def _read_points(s_or_path: str) -> np.ndarray:
    if os.path.exists(s_or_path):
        with open(s_or_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pts = np.asarray(data['points'], dtype=np.float32)
    else:
        pts = np.array([list(map(float, p.split(','))) for p in s_or_path.strip().split()], dtype=np.float32)
    assert pts.shape == (4, 2)
    return pts


def hstack_images(imgs):
    h = max(im.shape[0] for im in imgs)
    pads = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for im in imgs]
    return cv2.hconcat(pads)


def main():
    p = argparse.ArgumentParser(description="Demo pipeline: refine -> rectify (before/after)")
    p.add_argument("--image", required=True)
    p.add_argument("--points", required=True, help="init points JSON or string 'x1,y1 ... x4,y4'")
    p.add_argument("--method", default="edge", choices=["noop", "subpix", "edge", "find4quad"])
    p.add_argument("--out", default="results/pipeline")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    init = _read_points(args.points)

    refined, meta = refine_corners(img, init, CornerRefinerConfig(method=args.method))

    before_patch = warp_quad(img, init)
    after_patch = warp_quad(img, refined)

    vis = img.copy()
    vis = draw_quad(vis, init, (0, 0, 255), 2)
    vis = draw_quad(vis, refined, (0, 255, 0), 2)
    vis = draw_legend(vis, [("before", (0, 0, 255)), ("after", (0, 255, 0))])

    patches = hstack_images([before_patch, after_patch])

    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_{args.method}_overlay.png"), vis)
    cv2.imwrite(os.path.join(args.out, f"{stem}_{args.method}_patches.png"), patches)

    print(f"Saved overlay and rectified patches to {args.out}")


if __name__ == "__main__":
    main()

