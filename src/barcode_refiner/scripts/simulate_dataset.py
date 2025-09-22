from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import cv2
import numpy as np


def _ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)


def random_homography(w: int, h: int, max_perturb: float = 0.15) -> np.ndarray:
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    jitter = np.random.uniform(-max_perturb, max_perturb, size=(4, 2)).astype(np.float32)
    dst = src + np.stack([w, h], axis=0) * jitter
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def main():
    p = argparse.ArgumentParser(description="Create a simple synthetic dataset with init and gt quads")
    p.add_argument("--image", required=True, help="Source planar image (e.g., code sample)")
    p.add_argument("--count", type=int, default=20, help="Number of warped samples")
    p.add_argument("--out", default="data/sim", help="Output directory")
    p.add_argument("--noise", type=float, default=3.0, help="Pixel noise added to init corners")
    args = p.parse_args()

    _ensure_dir(args.out)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    h, w = img.shape[:2]

    base_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    for i in range(args.count):
        H = random_homography(w, h, 0.25)
        warped = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
        pts = cv2.perspectiveTransform(base_quad.reshape(1, 4, 2), H).reshape(4, 2)
        noise = np.random.normal(0, args.noise, size=(4, 2)).astype(np.float32)
        init = pts + noise

        stem = f"sim_{i:03d}"
        cv2.imwrite(os.path.join(args.out, stem + ".png"), warped)
        with open(os.path.join(args.out, stem + ".gt.json"), 'w', encoding='utf-8') as f:
            json.dump({"points": pts.tolist()}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, stem + ".init.json"), 'w', encoding='utf-8') as f:
            json.dump({"points": init.tolist()}, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.count} samples to {args.out}")


if __name__ == "__main__":
    main()

