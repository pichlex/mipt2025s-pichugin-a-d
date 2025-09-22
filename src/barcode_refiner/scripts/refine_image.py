from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import cv2
import numpy as np

from ..eval import parse_points_string, overlay_before_after, save_points
from ..refine import CornerRefinerConfig, refine_corners


def _ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description="Refine 4-point quad on an image and save visualization")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--points", required=True, help="'x1,y1 x2,y2 x3,y3 x4,y4' or a JSON/TXT file path")
    p.add_argument("--method", default="edge", choices=["noop", "subpix", "edge", "find4quad"], help="Refinement method")
    p.add_argument("--out", default="results/single", help="Output directory")
    p.add_argument("--edge-radius", type=int, default=4, help="Edge refiner: search radius (band half-width)")
    p.add_argument("--edge-canny-low", type=int, default=50, help="Edge refiner: Canny low threshold")
    p.add_argument("--edge-canny-high", type=int, default=150, help="Edge refiner: Canny high threshold")
    p.add_argument("--gt", default=None, help="Optional ground truth points file for IoU computation")
    args = p.parse_args()

    _ensure_dir(args.out)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    if os.path.exists(args.points):
        if args.points.lower().endswith('.json'):
            with open(args.points, 'r', encoding='utf-8') as f:
                init = np.asarray(json.load(f)['points'], dtype=np.float32)
        else:
            with open(args.points, 'r', encoding='utf-8') as f:
                init = parse_points_string(f.read())
    else:
        init = parse_points_string(args.points)

    cfg = CornerRefinerConfig(method=args.method, search_radius=args.edge_radius, canny_low=args.edge_canny_low, canny_high=args.edge_canny_high)
    refined, meta = refine_corners(img, init, cfg)

    gt = None
    iou_before = iou_after = None
    if args.gt:
        with open(args.gt, 'r', encoding='utf-8') as f:
            gt = np.asarray(json.load(f)['points'], dtype=np.float32)
        from ..geometry import quad_iou
        iou_before = quad_iou(init, gt)
        iou_after = quad_iou(refined, gt)

    vis = overlay_before_after(img, init, refined, gt)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_png = os.path.join(args.out, f"{stem}_{args.method}.png")
    out_json = os.path.join(args.out, f"{stem}_{args.method}.json")
    cv2.imwrite(out_png, vis)
    save_points(out_json, refined)

    report = {
        "image": args.image,
        "method": args.method,
        "meta": meta,
        "iou_before": iou_before,
        "iou_after": iou_after,
    }
    with open(os.path.join(args.out, f"{stem}_{args.method}.report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
