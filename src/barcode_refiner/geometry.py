from __future__ import annotations

import math
from typing import Iterable, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def order_corners(pts: Iterable[Iterable[float]]) -> np.ndarray:
    """Order points as tl, tr, br, bl for a convex quad."""
    p = np.asarray(pts, dtype=np.float32)
    if p.shape != (4, 2):
        raise ValueError(f"Expected 4x2 points, got shape {p.shape}")
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).flatten()
    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmin(d)]
    bl = p[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_size(pts: Iterable[Iterable[float]]) -> Tuple[int, int]:
    """Estimate rectified width and height from quad side lengths."""
    q = order_corners(pts)
    (tl, tr, br, bl) = q
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    width = int(max(widthA, widthB))
    height = int(max(heightA, heightB))
    width = max(width, 8)
    height = max(height, 8)
    return width, height


def get_homographies(pts: Iterable[Iterable[float]], size: Tuple[int, int]):
    q = order_corners(pts)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    Hinv = cv2.getPerspectiveTransform(dst, q.astype(np.float32))
    return H, Hinv


def warp_quad(image: np.ndarray, pts: Iterable[Iterable[float]], size: Tuple[int, int] | None = None) -> np.ndarray:
    if size is None:
        size = quad_size(pts)
    H, _ = get_homographies(pts, size)
    w, h = size
    return cv2.warpPerspective(image, H, (w, h))


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area; expects Nx2 order (closed not required)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Intersection of two lines (p1,p2) and (p3,p4)."""
    xdiff = np.array([p1[0] - p2[0], p3[0] - p4[0]])
    ydiff = np.array([p1[1] - p2[1], p3[1] - p4[1]])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if abs(div) < 1e-8:
        # Parallel lines; return midpoint between nearest endpoints
        return (p2 + p3) / 2.0
    d = np.array([det(p1, p2), det(p3, p4)])
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y], dtype=np.float32)


def sutherland_hodgman_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Clip convex subject polygon by convex clipper polygon. Returns polygon."""

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def compute_intersection(p1, p2, a, b):
        return _line_intersection(p1, p2, a, b)

    output = subject.copy()
    for i in range(len(clipper)):
        input_list = output
        output = []
        A = clipper[i]
        B = clipper[(i + 1) % len(clipper)]
        if isinstance(input_list, list):
            input_list = np.array(input_list, dtype=np.float32)
        if input_list is None or len(input_list) == 0:
            # Early terminate: empty polygon
            output = np.zeros((0, 2), dtype=np.float32)
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    inter = compute_intersection(S, E, A, B)
                    output.append(inter)
                output.append(E)
            elif inside(S, A, B):
                inter = compute_intersection(S, E, A, B)
                output.append(inter)
            S = E
        output = np.array(output, dtype=np.float32)
    if not isinstance(output, np.ndarray):
        output = np.array(output, dtype=np.float32)
    # Ensure empty returns have consistent shape
    if output.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return output


def quad_iou(q1: Iterable[Iterable[float]], q2: Iterable[Iterable[float]]) -> float:
    a = np.asarray(order_corners(q1), dtype=np.float32)
    b = np.asarray(order_corners(q2), dtype=np.float32)
    inter_poly = sutherland_hodgman_clip(a, b)
    inter_poly = np.asarray(inter_poly, dtype=np.float32)
    if inter_poly.size == 0 or inter_poly.shape[0] < 3:
        return 0.0
    inter_area = abs(polygon_area(inter_poly))
    a_area = abs(polygon_area(a))
    b_area = abs(polygon_area(b))
    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def draw_quad(img: np.ndarray, pts: Iterable[Iterable[float]], color=(0, 0, 255), thickness=2) -> np.ndarray:
    vis = img.copy()
    q = order_corners(pts).astype(int)
    for i in range(4):
        p1 = tuple(q[i])
        p2 = tuple(q[(i + 1) % 4])
        cv2.line(vis, p1, p2, color, thickness, cv2.LINE_AA)
    return vis


def draw_quad_filled(
    img: np.ndarray,
    pts: Iterable[Iterable[float]],
    edge_color=(0, 0, 255),
    fill_color=(0, 0, 255),
    alpha: float = 0.25,
    thickness: int = 2,
    draw_indices: bool = False,
) -> np.ndarray:
    """Draw quad with semi-transparent fill and optional corner indices."""
    vis = img.copy()
    q = order_corners(pts).astype(np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [q], fill_color)
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    for i in range(4):
        p1 = tuple(q[i])
        p2 = tuple(q[(i + 1) % 4])
        cv2.line(vis, p1, p2, edge_color, thickness, cv2.LINE_AA)
    if draw_indices:
        for i, (x, y) in enumerate(q):
            cv2.circle(vis, (int(x), int(y)), 3, edge_color, -1)
            cv2.putText(vis, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge_color, 1, cv2.LINE_AA)
    return vis


def draw_legend(img: np.ndarray, entries: list, origin: tuple = (10, 10), box_alpha: float = 0.6) -> np.ndarray:
    """Draw legend box with colored squares and labels."""
    vis = img.copy()
    x, y = origin
    pad = 6
    sw, sh = 14, 14  # square size
    line_h = max(18, sh + 4)
    width = 0
    # compute width
    for label, _ in entries:
        width = max(width, 8 + sw + 8 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0])
    height = pad * 2 + line_h * len(entries)
    # background box
    bg = vis.copy()
    cv2.rectangle(bg, (x, y), (x + width + pad * 2, y + height), (255, 255, 255), -1)
    vis = cv2.addWeighted(bg, box_alpha, vis, 1 - box_alpha, 0)
    # draw entries
    cy = y + pad + sh
    for label, color in entries:
        cv2.rectangle(vis, (x + pad, cy - sh), (x + pad + sw, cy), color, -1)
        cv2.putText(vis, label, (x + pad + sw + 8, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cy += line_h
    return vis

