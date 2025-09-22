from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..geometry import warp_quad


def _box_points(rect) -> np.ndarray:
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32)


def _rect_area(rect) -> float:
    (_, _), (w, h), _ = rect
    return float(w) * float(h)


def _contour_area(cnt) -> float:
    return float(cv2.contourArea(cnt))


def _ratio(a: float, b: float) -> float:
    if b <= 1e-6:
        return 0.0
    return float(a / b)


def _approx_quad(cnt, eps_frac=0.02) -> np.ndarray | None:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
    if approx is None:
        return None
    if len(approx) != 4:
        return None
    quad = approx.reshape(4, 2).astype(np.float32)
    return quad


def _valid_quad(quad: np.ndarray, min_size: int = 12) -> bool:
    if quad is None or quad.shape != (4, 2):
        return False
    # check bbox size
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    return (w >= min_size and h >= min_size)


def detect_1d_barcodes(image: np.ndarray, max_candidates: int = 8) -> List[np.ndarray]:
    """Heuristic 1D barcode detector using gradients + morphology.

    Returns list of 4x2 quads in image coordinates.
    """
    img = image
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # strong x-gradient highlights bars
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradX = cv2.convertScaleAbs(gradX)
    # normalize and threshold
    gradX = cv2.normalize(gradX, None, 0, 255, cv2.NORM_MINMAX)
    _, bw = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # close gaps along x direction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    # opening to reduce noise
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    # find contours
    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter by aspect ratio and fill ratio
    cand: List[Tuple[float, np.ndarray]] = []
    H, W = gray.shape[:2]
    area_img = H * W
    for c in cnts:
        if len(c) < 5:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        if w < 8 or h < 8:
            continue
        big = max(w, h)
        small = min(w, h)
        ar = _ratio(big, small)
        if ar < 2.0:  # barcode shape tends to be elongated
            continue
        area_r = _rect_area(rect)
        area_c = _contour_area(c)
        fill = _ratio(area_c, area_r)
        if fill < 0.25:  # too sparse
            continue
        if area_r < 0.0005 * area_img:
            continue
        quad = _box_points(rect)
        if not _valid_quad(quad, 12):
            continue
        score = area_r * ar * fill
        cand.append((score, quad))

    cand.sort(key=lambda x: x[0], reverse=True)
    out = [q for _, q in cand[:max_candidates]]
    return out


def detect_2d_codes(image: np.ndarray, max_candidates: int = 6) -> List[np.ndarray]:
    """Heuristic 2D code detector via binarization + contour quads + QR validation."""
    img = image
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold to capture squares in varying lighting
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    bw = cv2.bitwise_not(bw)
    # remove tiny noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape[:2]
    area_img = H * W
    qr = cv2.QRCodeDetector()
    cand: List[Tuple[float, np.ndarray]] = []
    for c in cnts:
        if len(c) < 4:
            continue
        quad = _approx_quad(c, 0.03)
        if quad is None:
            continue
        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        if min(w, h) < 16:  # too small
            continue
        area = float(w * h)
        if area < 0.0008 * area_img:
            continue
        ratio = _ratio(max(w, h), min(w, h))
        if ratio > 1.6:  # prefer near-squares
            continue
        # validate with QR detector on rectified patch
        try:
            patch = warp_quad(img, quad)
            val, _, _ = qr.detectAndDecode(patch)
            ok = bool(val)
        except Exception:
            ok = False
        score = area * (1.0 / max(1.0, ratio)) * (2.0 if ok else 1.0)
        cand.append((score, quad))

    cand.sort(key=lambda x: x[0], reverse=True)
    out = [q for _, q in cand[:max_candidates]]
    return out


class HeuristicDetector:
    """A robust fallback detector combining 1D and 2D heuristics.

    Intended to provide candidates when learned/stock detectors fail.
    """

    def __init__(self) -> None:
        pass

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        q1 = detect_1d_barcodes(image)
        q2 = detect_2d_codes(image)
        return q1 + q2

