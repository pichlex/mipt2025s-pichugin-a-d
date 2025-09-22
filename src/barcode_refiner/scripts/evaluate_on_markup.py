from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import List, Dict

import cv2
import numpy as np

from ..refine import refine_corners, CornerRefinerConfig
from ..geometry import quad_iou, draw_quad, draw_legend
from ..logging_utils import setup_logging, get_progress


def _load_markup(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expecting { objects: [ {type: 'quad', data: [[x,y]..4], tags: [...] }, ... ] }
    objs = data.get('objects', [])
    quads = []
    for o in objs:
        if o.get('type') != 'quad':
            continue
        pts = np.asarray(o.get('data'), dtype=np.float32)
        if pts.shape == (4, 2):
            quads.append({"points": pts, "tags": o.get('tags', [])})
    return quads


def _noisy_init(pts: np.ndarray, noise: float) -> np.ndarray:
    # noise is in pixels (stddev)
    return pts + np.random.normal(0, noise, size=pts.shape).astype(np.float32)


def run_markup_eval(images_dir: str, markup_dir: str, out: str, methods: List[str], noise: float, *, verbose: int = 1, progress: bool = True) -> Dict[str, Dict]:
    logger = logging.getLogger(__name__)
    os.makedirs(out, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    images.sort()
    logger.info(f"Found {len(images)} images in {images_dir}")

    all_results: Dict[str, Dict] = {}
    for method in get_progress(methods, desc="methods", total=len(methods), progress=progress):
        run_out = os.path.join(out, method)
        os.makedirs(run_out, exist_ok=True)
        details = []
        logger.info(f"Start method='{method}' -> {run_out}")
        t0 = time.perf_counter()

        for name in get_progress(images, desc=f"{method}", total=len(images), progress=progress):
            stem, ext = os.path.splitext(name)
            img_path = os.path.join(images_dir, name)
            markup_path = os.path.join(markup_dir, stem + ext + ".json")
            if not os.path.exists(markup_path):
                alt = os.path.join(markup_dir, stem + ".json")
                if os.path.exists(alt):
                    markup_path = alt
            if not os.path.exists(markup_path):
                logger.debug(f"Skip image without markup: {name}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            quads = _load_markup(markup_path)
            if verbose >= 2:
                logger.debug(f"{name}: {len(quads)} quads from markup")
            for qi, q in enumerate(quads):
                gt = q["points"].astype(np.float32)
                init = _noisy_init(gt, noise)
                refined, meta = refine_corners(img, init, CornerRefinerConfig(method=method))
                iou_b = quad_iou(init, gt)
                iou_a = quad_iou(refined, gt)
                rec = {
                    "image": name,
                    "index": qi,
                    "method": method,
                    "iou_before": float(iou_b),
                    "iou_after": float(iou_a),
                    "iou_gain": float(iou_a - iou_b),
                }
                details.append(rec)
                vis = img.copy()
                vis = draw_quad(vis, gt, (255, 0, 0), 2)
                vis = draw_quad(vis, init, (0, 0, 255), 2)
                vis = draw_quad(vis, refined, (0, 255, 0), 2)
                vis = draw_legend(vis, [("gt", (255, 0, 0)), ("init", (0, 0, 255)), ("refined", (0, 255, 0))])
                cv2.imwrite(os.path.join(run_out, f"{stem}_{qi:02d}.png"), vis)

        gains = [d["iou_gain"] for d in details]
        summary = {
            "count": len(details),
            "avg_iou_gain": float(np.mean(gains)) if gains else None,
            "median_iou_gain": float(np.median(gains)) if gains else None,
            "details": details,
        }
        with open(os.path.join(run_out, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        all_results[method] = summary
        dt = time.perf_counter() - t0
        logger.info(f"Done method='{method}': {len(details)} items in {dt:.2f}s (avg {dt / max(1, len(details)):.3f}s/item)")

    with open(os.path.join(out, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results


def main():
    p = argparse.ArgumentParser(description="Evaluate refiners on a dataset with markup/*.json (objects: quads)")
    p.add_argument("--images_dir", required=True)
    p.add_argument("--markup_dir", required=True)
    p.add_argument("--out", default="results/markup_eval")
    p.add_argument("--methods", nargs="*", default=["noop", "subpix", "edge"], help="Methods to compare")
    p.add_argument("--noise", type=float, default=3.0, help="Stddev of pixel noise added to GT to form init")
    p.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = p.parse_args()

    setup_logging(args.verbose)
    res = run_markup_eval(args.images_dir, args.markup_dir, args.out, args.methods, args.noise, verbose=args.verbose, progress=not args.no_progress)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
