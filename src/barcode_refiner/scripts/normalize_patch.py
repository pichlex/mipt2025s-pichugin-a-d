from __future__ import annotations

import argparse
import json
import os
import cv2
import numpy as np

from ..geometry import warp_quad


def main():
    p = argparse.ArgumentParser(description="Rectify a quad region from image via perspective warp")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--points", required=True, help="JSON with {points: [[x,y],...]} or 'x1,y1 x2,y2 x3,y3 x4,y4'")
    p.add_argument("--out", default="results/rectify", help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    if os.path.exists(args.points):
        with open(args.points, 'r', encoding='utf-8') as f:
            pts = np.asarray(json.load(f)['points'], dtype=np.float32)
    else:
        parts = args.points.strip().split()
        pts = np.array([list(map(float, p.split(','))) for p in parts], dtype=np.float32)

    patch = warp_quad(img, pts)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_rectified.png"), patch)
    print(os.path.join(args.out, f"{stem}_rectified.png"))


if __name__ == "__main__":
    main()

