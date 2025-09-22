from __future__ import annotations

"""
Local corner auto-refiner for projectively distorted regions.

Public API:
    refine_corners(image, points, config) -> (refined_points, meta)

Methods:
 - 'noop'      : returns points unchanged
 - 'subpix'    : warp to rect, run cornerSubPix near rect corners, unwarp back
 - 'edge'      : fit lines in narrow bands along sides; corners as intersections
 - 'find4quad' : try cv2.find4QuadCornerSubpix (if available)
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict

import cv2
import numpy as np

from .geometry import order_corners, get_homographies, quad_size


@dataclass
class CornerRefinerConfig:
    window_size: int = 21
    max_iters: int = 20
    search_radius: int = 4
    canny_low: int = 50
    canny_high: int = 150
    method: str = "subpix"  # noop | subpix | edge | find4quad
    debug: bool = False


def _to_numpy_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4x2 points, got shape {pts.shape}")
    return pts


def refine_corners(
    image: np.ndarray,
    points: Iterable[Iterable[float]],
    config: Optional[CornerRefinerConfig] = None,
) -> Tuple[np.ndarray, Dict]:
    if config is None:
        config = CornerRefinerConfig()
    pts = _to_numpy_points(points)
    method = (config.method or "subpix").lower()

    if method == "noop":
        return pts, {"status": "noop", "method": method}
    if method == "subpix":
        return _refine_subpix(image, pts, config)
    if method == "edge":
        return _refine_edges(image, pts, config)
    if method == "find4quad":
        try:
            return _refine_find4quad(image, pts, config)
        except Exception as e:
            refined, meta = _refine_subpix(image, pts, config)
            meta["fallback"] = f"find4quad failed: {e}"
            return refined, meta
    # default fallback
    refined, meta = _refine_subpix(image, pts, config)
    meta["fallback"] = f"unknown method '{method}', used subpix"
    return refined, meta


def visualize_corners(image: np.ndarray, points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = _to_numpy_points(points).astype(int)
    vis = image.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 3, colors[i % len(colors)], thickness=-1)
        cv2.putText(vis, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1, cv2.LINE_AA)
    return vis


def _refine_subpix(image: np.ndarray, pts: np.ndarray, config: CornerRefinerConfig) -> Tuple[np.ndarray, Dict]:
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        size = quad_size(pts)
    except Exception:
        size = (max(gray.shape[1] // 8, 32), max(gray.shape[0] // 8, 32))

    H, Hinv = get_homographies(pts, size)
    warped = cv2.warpPerspective(gray, H, size)
    if warped.ndim == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)

    w, h = size
    corners = np.array([
        [2.0, 2.0],
        [w - 3.0, 2.0],
        [w - 3.0, h - 3.0],
        [2.0, h - 3.0],
    ], dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max(10, config.max_iters),
        0.01,
    )
    try:
        cv2.cornerSubPix(
            warped.astype(np.float32),
            corners.reshape(-1, 1, 2),
            (config.window_size, config.window_size),
            (-1, -1),
            criteria,
        )
    except Exception:
        pass

    ones = np.ones((4, 1), dtype=np.float32)
    pts_h = np.concatenate([corners, ones], axis=1).T  # 3x4
    refined_h = (Hinv @ pts_h)
    refined = (refined_h[:2, :] / refined_h[2:3, :]).T.astype(np.float32)

    meta = {"status": "ok", "method": "subpix"}
    return refined, meta


def _fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("Not enough points to fit a line")
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    origin = np.array([float(x0), float(y0)], dtype=np.float32)
    return origin, direction


def _line_from_two_points(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        d = np.array([1.0, 0.0], dtype=np.float32)
    else:
        d = d / norm
    return p1.astype(np.float32), d.astype(np.float32)


def _intersect_lines(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    A = np.array([d1, -d2]).T
    b = o2 - o1
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return (o1 + o2) / 2.0
    t = np.linalg.solve(A, b)[0]
    return o1 + t * d1


def _refine_edges(image: np.ndarray, pts: np.ndarray, config: CornerRefinerConfig) -> Tuple[np.ndarray, Dict]:
    import math

    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny thresholds are configurable via CornerRefinerConfig
    low = int(max(0, config.canny_low))
    high = int(max(low + 1, config.canny_high))
    edges = cv2.Canny(gray, low, high)

    q = order_corners(pts)
    lines = []
    band_width = max(4, config.search_radius * 2)
    for i in range(4):
        p1 = q[i]
        p2 = q[(i + 1) % 4]
        center = (p1 + p2) / 2.0
        length = float(np.linalg.norm(p2 - p1))
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        rect = ((float(center[0]), float(center[1])), (length, float(band_width)), float(angle))
        box = cv2.boxPoints(rect).astype(np.int32)
        mask = np.zeros_like(edges, dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        ys, xs = np.where((edges > 0) & (mask > 0))
        pts_band = np.stack([xs, ys], axis=1) if len(xs) else np.empty((0, 2), dtype=np.float32)
        if pts_band.shape[0] >= 20:
            o, d = _fit_line(pts_band)
        else:
            o, d = _line_from_two_points(p1, p2)
        lines.append((o, d))

    refined = []
    for i in range(4):
        o1, d1 = lines[i]
        o2, d2 = lines[(i + 3) % 4]
        inter = _intersect_lines(o1, d1, o2, d2)
        refined.append(inter)
    refined = np.array(refined, dtype=np.float32)
    refined = order_corners(refined)
    meta = {"status": "ok", "method": "edge"}
    return refined, meta


def _refine_find4quad(image: np.ndarray, pts: np.ndarray, config: CornerRefinerConfig) -> Tuple[np.ndarray, Dict]:
    gray = image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    q = order_corners(pts)
    corners = q.copy().reshape(-1, 1, 2)
    win = max(5, config.window_size)
    func = getattr(cv2, 'find4QuadCornerSubpix', None)
    if func is None:
        raise RuntimeError('cv2.find4QuadCornerSubpix not available')
    ok = func(gray, corners, win)
    if not ok:
        raise RuntimeError('find4QuadCornerSubpix failed to refine corners')
    refined = corners.reshape(4, 2).astype(np.float32)
    refined = order_corners(refined)
    meta = {"status": "ok", "method": "find4quad"}
    return refined, meta
