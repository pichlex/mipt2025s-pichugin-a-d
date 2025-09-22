from __future__ import annotations

import argparse
import json
import os
import cv2
import numpy as np

from ..refine import CornerRefinerConfig, refine_corners
from ..geometry import warp_quad


def _read_points(s_or_path: str) -> np.ndarray:
    if os.path.exists(s_or_path):
        with open(s_or_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pts = np.asarray(data['points'], dtype=np.float32)
    else:
        pts = np.array([list(map(float, p.split(','))) for p in s_or_path.strip().split()], dtype=np.float32)
    assert pts.shape == (4, 2)
    return pts


def main():
    p = argparse.ArgumentParser(description="Decode QR on rectified patch before/after refinement and compare")
    p.add_argument("--image", required=True)
    p.add_argument("--points", required=True, help="init points JSON or 'x1,y1 ... x4,y4'")
    p.add_argument("--method", default="edge", choices=["noop", "subpix", "edge", "find4quad"])
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    init = _read_points(args.points)

    refined, _ = refine_corners(img, init, CornerRefinerConfig(method=args.method))

    before = warp_quad(img, init)
    after = warp_quad(img, refined)

    qr = cv2.QRCodeDetector()
    val_b, _, _ = qr.detectAndDecode(before)
    val_a, _, _ = qr.detectAndDecode(after)
    print(json.dumps({
        "method": args.method,
        "decoded_before": bool(val_b),
        "decoded_after": bool(val_a),
        "value_before": val_b,
        "value_after": val_a,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

