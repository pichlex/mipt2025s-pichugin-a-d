from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import cv2
import numpy as np

from .geometry import order_corners, quad_iou, draw_quad, draw_quad_filled, draw_legend
from .refine import refine_corners, CornerRefinerConfig


def parse_points_string(s: str) -> np.ndarray:
    parts = s.strip().split()
    pts = []
    for p in parts:
        x, y = p.split(',')
        pts.append([float(x), float(y)])
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected 4 comma-separated pairs: 'x1,y1 x2,y2 x3,y3 x4,y4'")
    return pts


def load_points(path: str) -> np.ndarray:
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pts = np.asarray(data['points'], dtype=np.float32)
    else:
        # txt with space-separated pairs
        with open(path, 'r', encoding='utf-8') as f:
            s = f.read().strip()
        pts = parse_points_string(s)
    if pts.shape != (4, 2):
        raise ValueError(f"Bad points shape in {path}: {pts.shape}")
    return pts


def save_points(path: str, pts: Iterable[Iterable[float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"points": pts.tolist()}, f, ensure_ascii=False, indent=2)


def overlay_before_after(img: np.ndarray, before: Iterable[Iterable[float]], after: Iterable[Iterable[float]], gt: Iterable[Iterable[float]] | None = None) -> np.ndarray:
    vis = img.copy()
    vis = draw_quad(vis, before, (0, 0, 255), 2)
    vis = draw_quad(vis, after, (0, 255, 0), 2)
    entries = [("before", (0, 0, 255)), ("after", (0, 255, 0))]
    if gt is not None:
        vis = draw_quad(vis, gt, (255, 0, 0), 2)
        entries.insert(0, ("gt", (255, 0, 0)))
    vis = draw_legend(vis, entries)
    return vis


def evaluate_on_dir(
    images_dir: str,
    init_suffix: str = ".init.json",
    gt_suffix: str | None = ".gt.json",
    method: str = "edge",
    out_dir: str | None = None,
) -> Dict:
    os.makedirs(out_dir or "results/eval", exist_ok=True)
    stats: List[Dict] = []
    for name in os.listdir(images_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(images_dir, name)
        init_path = os.path.join(images_dir, stem + init_suffix)
        if not os.path.exists(init_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        init = load_points(init_path)
        cfg = CornerRefinerConfig(method=method)
        refined, meta = refine_corners(img, init, cfg)
        rec: Dict = {"image": name, "method": method}
        if gt_suffix:
            gt_path = os.path.join(images_dir, stem + gt_suffix)
            if os.path.exists(gt_path):
                gt = load_points(gt_path)
                iou_before = quad_iou(init, gt)
                iou_after = quad_iou(refined, gt)
                rec.update({"iou_before": iou_before, "iou_after": iou_after, "iou_gain": iou_after - iou_before})
                vis = overlay_before_after(img, init, refined, gt)
            else:
                vis = overlay_before_after(img, init, refined)
        else:
            vis = overlay_before_after(img, init, refined)
        if out_dir:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{method}.png"), vis)
            save_points(os.path.join(out_dir, f"{stem}_{method}.json"), refined)
        stats.append(rec)
    # aggregate
    gains = [s.get("iou_gain", 0.0) for s in stats if "iou_gain" in s]
    summary = {
        "count": len(stats),
        "avg_iou_gain": float(np.mean(gains)) if gains else None,
        "median_iou_gain": float(np.median(gains)) if gains else None,
        "details": stats,
    }
    if out_dir:
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

