from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from ..geometry import order_corners, quad_iou


def _rotate_points(q: np.ndarray, shape: Tuple[int, int], k: int) -> np.ndarray:
    H, W = shape
    p = q.copy()
    if k % 4 == 0:
        return p
    if k % 4 == 1:
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = W - 1 - y
        p[:, 1] = x
        return p
    if k % 4 == 2:
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = W - 1 - x
        p[:, 1] = H - 1 - y
        return p
    if k % 4 == 3:
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = y
        p[:, 1] = H - 1 - x
        return p
    return p


def _nms_quads(quads: List[np.ndarray], scores: List[float], iou_thr: float = 0.5) -> List[np.ndarray]:
    idxs = list(range(len(quads)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    used = [False] * len(quads)
    for i in idxs:
        if used[i]:
            continue
        keep.append(i)
        for j in idxs:
            if used[j] or j == i:
                continue
            if quad_iou(quads[i], quads[j]) >= iou_thr:
                used[j] = True
    return [quads[i] for i in keep]


class AutoDetector:
    """Detector built from scratch (no external weights) for QR and 1D barcodes.

    Strategy:
      - Work at reduced resolution for speed (keeps scale factors).
      - Try 4 rotations (0/90/180/270).
      - 1D: gradient->threshold->morphology->contours->minAreaRect, filter by aspect/fill.
      - 2D: adaptive thresh/Canny->contours->approxPolyDP to quads, filter by squareness; validate with QRCodeDetector on patch when possible.
      - Merge candidates and apply NMS on quads.
    """

    def __init__(self, max_side: int = 1024) -> None:
        self.max_side = int(max_side)
        self.qr = cv2.QRCodeDetector()

    def _resize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        s = 1.0
        if max(h, w) > self.max_side:
            s = self.max_side / float(max(h, w))
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img, s

    def _detect_1d(self, img: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        g = cv2.GaussianBlur(g, (3, 3), 0)
        grad_x = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
        _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Try multiple horizontal kernels to close gaps
        kernels = [(21, 3), (31, 3), (41, 3), (51, 3)]
        cand = []
        H, W = g.shape[:2]
        area_img = H * W
        for kx, ky in kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
            morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if len(c) < 5:
                    continue
                rect = cv2.minAreaRect(c)
                (cx, cy), (w, h), ang = rect
                if w < 8 or h < 8:
                    continue
                big, small = (w, h) if w >= h else (h, w)
                ar = (big + 1e-6) / (small + 1e-6)
                if ar < 2.2:
                    continue
                area_r = float(w * h)
                area_c = float(cv2.contourArea(c))
                if area_r < 0.0005 * area_img:
                    continue
                fill = area_c / (area_r + 1e-6)
                if fill < 0.25:
                    continue
                box = cv2.boxPoints(rect).astype(np.float32)
                out.append(box)
        return out

    def _detect_2d(self, img: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        g = cv2.GaussianBlur(g, (3, 3), 0)
        # Use both adaptive and Canny
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        bw = cv2.bitwise_not(bw)
        edges = cv2.Canny(g, 60, 180)
        mask = cv2.bitwise_or(bw, edges)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = g.shape[:2]
        area_img = H * W
        for c in cnts:
            if len(c) < 4:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if approx is None or len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
            if min(w, h) < 12:
                continue
            area = float(w * h)
            if area < 0.0008 * area_img:
                continue
            # near-square check
            ratio = (max(w, h) + 1e-6) / (min(w, h) + 1e-6)
            if ratio > 1.8:
                continue
            # optional QR validation on patch
            try:
                patch_w = max(24, int(max(w, h)))
                M = cv2.getPerspectiveTransform(order_corners(quad), np.array([[0, 0], [patch_w - 1, 0], [patch_w - 1, patch_w - 1], [0, patch_w - 1]], dtype=np.float32))
                patch = cv2.warpPerspective(img, M, (patch_w, patch_w))
                val, _, _ = self.qr.detectAndDecode(patch)
                if val or ratio < 1.2:
                    out.append(quad)
            except Exception:
                out.append(quad)
        return out

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        if image is None:
            return []
        H0, W0 = image.shape[:2]
        small, s = self._resize(image)
        Hs, Ws = small.shape[:2]
        candidates: List[np.ndarray] = []
        scores: List[float] = []
        # try 4 rotations
        for k in (0, 1, 2, 3):
            if k == 0:
                img = small
            else:
                img = np.rot90(small, k)
            # 1D
            one_d = self._detect_1d(img)
            for q in one_d:
                q = _rotate_points(q, (Hs, Ws), 4 - (k % 4))
                q = q / s
                candidates.append(q)
                x, y, w, h = cv2.boundingRect(q.astype(np.int32))
                ar = (max(w, h) + 1e-6) / (min(w, h) + 1e-6)
                scores.append(float(w * h) * ar)
            # 2D
            two_d = self._detect_2d(img)
            for q in two_d:
                q = _rotate_points(q, (Hs, Ws), 4 - (k % 4))
                q = q / s
                x, y, w, h = cv2.boundingRect(q.astype(np.int32))
                ar = (max(w, h) + 1e-6) / (min(w, h) + 1e-6)
                candidates.append(q)
                scores.append(float(w * h) * (1.5 - min(ar, 1.5)))

        # clip and sanitize
        out = []
        for q in candidates:
            q[:, 0] = np.clip(q[:, 0], 0, W0 - 1)
            q[:, 1] = np.clip(q[:, 1], 0, H0 - 1)
            out.append(order_corners(q))

        if not out:
            return []
        # NMS on quads
        out = _nms_quads(out, scores, iou_thr=0.5)
        return [q.astype(np.float32) for q in out]

