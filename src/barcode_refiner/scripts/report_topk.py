from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List

import cv2


def tile_images(paths: List[str], cols: int = 3, scale: float = 1.0):
    imgs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    # resize
    if abs(scale - 1.0) > 1e-3:
        imgs = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in imgs]
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    # pad to same size
    norm = []
    for im in imgs:
        pad = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        norm.append(pad)
    rows = (len(norm) + cols - 1) // cols
    grid = []
    for r in range(rows):
        row = norm[r * cols:(r + 1) * cols]
        if not row:
            break
        if len(row) < cols:
            # pad with blanks
            blank = 255 * (norm[0][..., :3] * 0).astype(norm[0].dtype)
            row += [blank] * (cols - len(row))
        grid.append(cv2.hconcat(row))
    return cv2.vconcat(grid)


def main():
    ap = argparse.ArgumentParser(description="Collect top-K best/worst overlays by IoU gain")
    ap.add_argument("--results", required=True, help="Root results dir with per-method subfolders")
    ap.add_argument("--method", default="edge", help="Method to analyze (subfolder name)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    method_dir = os.path.join(args.results, args.method)
    summ_path = os.path.join(method_dir, "summary.json")
    if not os.path.exists(summ_path):
        raise SystemExit(f"summary.json not found under {method_dir}")
    with open(summ_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    details = summary.get("details", [])
    # sort by iou_gain desc
    details_sorted = sorted(details, key=lambda d: d.get("iou_gain", 0.0), reverse=True)
    best = details_sorted[:args.k]
    worst = list(reversed(details_sorted))[:args.k]

    def overlay_path(rec):
        stem = os.path.splitext(rec["image"])[0]
        qi = rec["index"]
        return os.path.join(method_dir, f"{stem}_{qi:02d}.png")

    out_base = os.path.join(args.results, "topk", args.method)
    best_dir = os.path.join(out_base, "best")
    worst_dir = os.path.join(out_base, "worst")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    best_paths = []
    for i, rec in enumerate(best):
        src = overlay_path(rec)
        dst = os.path.join(best_dir, f"{i:02d}_{os.path.basename(src)}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            best_paths.append(dst)

    worst_paths = []
    for i, rec in enumerate(worst):
        src = overlay_path(rec)
        dst = os.path.join(worst_dir, f"{i:02d}_{os.path.basename(src)}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            worst_paths.append(dst)

    # grids
    grid_best = tile_images(best_paths, cols=args.cols, scale=1.0)
    grid_worst = tile_images(worst_paths, cols=args.cols, scale=1.0)
    if grid_best is not None:
        cv2.imwrite(os.path.join(out_base, f"best_{args.k}.png"), grid_best)
    if grid_worst is not None:
        cv2.imwrite(os.path.join(out_base, f"worst_{args.k}.png"), grid_worst)
    print(json.dumps({
        "results": args.results,
        "method": args.method,
        "best_count": len(best_paths),
        "worst_count": len(worst_paths)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

