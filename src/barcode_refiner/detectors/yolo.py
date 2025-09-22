from __future__ import annotations

from typing import List, Tuple, Optional

import os
import numpy as np
import cv2

from ..geometry import order_corners, quad_iou


def _rot_back_points(pts: np.ndarray, shape: Tuple[int, int], k: int) -> np.ndarray:
    H, W = shape
    p = pts.copy()
    k = k % 4
    if k == 0:
        return p
    if k == 1:  # 90 CCW applied to image -> invert: -90
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = W - 1 - y
        p[:, 1] = x
        return p
    if k == 2:
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = W - 1 - x
        p[:, 1] = H - 1 - y
        return p
    if k == 3:
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = y
        p[:, 1] = H - 1 - x
        return p
    return p


def _nms_quads(quads: List[np.ndarray], scores: List[float], iou_thr: float = 0.5) -> List[np.ndarray]:
    if not quads:
        return []
    idxs = list(range(len(quads)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    suppressed = [False] * len(quads)
    for i in idxs:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in idxs:
            if suppressed[j] or j == i:
                continue
            if quad_iou(quads[i], quads[j]) >= iou_thr:
                suppressed[j] = True
    return [quads[i] for i in keep]


class YOLODetector:
    """Ultralytics YOLO detector with rotation TTA and quad outputs.

    - weights: path to .pt weights (fine-tuned for barcodes/QR)
    - device: auto|mps|cuda|cpu
    - imgsz: int size for inference (max side)
    - conf: confidence threshold
    - iou: NMS IoU threshold (inside YOLO)
    - max_det: max detections per image
    - tta_rot: if True, run TTA over rotations [0, 90, 180, 270]
    - classes: list of class indices to keep (optional)
    """

    def __init__(
        self,
        weights: str = "YOLOV8s_Barcode_Detection.pt",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        tta_rot: bool = True,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.weights = weights
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.tta_rot = bool(tta_rot)
        self.classes = classes
        self.device = self._select_device(device)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. `uv pip install ultralytics`.") from e
        # Let Ultralytics handle auto-download for known model names/URLs
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            # Fallback to a public base model if custom string not resolvable
            self.model = YOLO("yolov8s.pt")
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def _select_device(self, device: str) -> str:
        if device and device != "auto":
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _predict(self, img: np.ndarray):
        # Use the same API whether via __call__ or .predict
        return self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )

    def _mask_polys_to_quads(self, masks_obj, H: int, W: int) -> List[np.ndarray]:
        quads: List[np.ndarray] = []
        try:
            # Ultralytics exposes polygons in masks.xy (list of Nx2 arrays)
            polys = getattr(masks_obj, "xy", None)
            if polys is None:
                # Fallback: raster masks -> contours
                data = getattr(masks_obj, "data", None)
                if data is not None:
                    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.array(data)
                    for m in arr:
                        m8 = (m * 255).astype(np.uint8)
                        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not cnts:
                            continue
                        c = max(cnts, key=cv2.contourArea)
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if approx is not None and len(approx) == 4:
                            q = approx.reshape(4, 2).astype(np.float32)
                        else:
                            rect = cv2.minAreaRect(c)
                            q = cv2.boxPoints(rect).astype(np.float32)
                        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
                        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
                        quads.append(order_corners(q))
                return quads
            # polygons available
            for p in polys:
                if p is None or len(p) < 3:
                    continue
                cnt = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if approx is not None and len(approx) == 4:
                    q = approx.reshape(4, 2).astype(np.float32)
                else:
                    rect = cv2.minAreaRect(cnt)
                    q = cv2.boxPoints(rect).astype(np.float32)
                q[:, 0] = np.clip(q[:, 0], 0, W - 1)
                q[:, 1] = np.clip(q[:, 1], 0, H - 1)
                quads.append(order_corners(q))
        except Exception:
            pass
        return quads

    def _roi_to_quad(self, img: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy[:4]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1] - 1, x2)
        y2 = min(img.shape[0] - 1, y2)
        roi = img[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0:
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        q = cv2.boxPoints(rect).astype(np.float32)
        q[:, 0] += x1
        q[:, 1] += y1
        return order_corners(q)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        H, W = image.shape[:2]
        rotations = [0, 1, 2, 3] if self.tta_rot else [0]
        all_quads: List[np.ndarray] = []
        scores: List[float] = []
        for k in rotations:
            if k == 0:
                img = image
                rot_shape = (H, W)
            else:
                img = np.rot90(image, k)
                rot_shape = img.shape[:2]
            res = self._predict(img)
            if not res:
                continue
            r = res[0]
            # Prefer segmentation masks if available to get edges/quads
            masks = getattr(r, "masks", None)
            if masks is not None:
                quads_m = self._mask_polys_to_quads(masks, rot_shape[0], rot_shape[1])
                for q in quads_m:
                    if k != 0:
                        q = _rot_back_points(q, (rot_shape[0], rot_shape[1]), k)
                    all_quads.append(order_corners(q))
                    scores.append(1.0)
            # Also gather boxes (useful fallback) and convert to edge-like quads via ROI refinement
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.ones((xyxy.shape[0],), dtype=float)
                except Exception:
                    xyxy, confs = None, None
                if xyxy is not None:
                    for (x1, y1, x2, y2, *_), sc in zip(xyxy, confs):
                        box = np.array([x1, y1, x2, y2], dtype=np.float32)
                        if k != 0:
                            # rotate corners of axis-aligned box, then refine in roi after rot-back
                            q_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                            q_box = _rot_back_points(q_box, (rot_shape[0], rot_shape[1]), k)
                            x1b, y1b = q_box.min(axis=0)
                            x2b, y2b = q_box.max(axis=0)
                            box = np.array([x1b, y1b, x2b, y2b], dtype=np.float32)
                        q = self._roi_to_quad(image, box)
                        all_quads.append(order_corners(q))
                        scores.append(float(sc))
        # NMS over aggregated quads
        quads_nms = _nms_quads(all_quads, scores, iou_thr=0.5)
        return [q.astype(np.float32) for q in quads_nms]
