from __future__ import annotations

from typing import List, Optional, Tuple

import os
import logging
import numpy as np
import cv2

from ..geometry import order_corners


def _select_device(device: str) -> str:
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


def _center_point(box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return np.array([[cx, cy]], dtype=np.float32)


def _mask_to_quad(mask: np.ndarray) -> Optional[np.ndarray]:
    """Approximate a boolean mask with a quad using contour approx or minAreaRect."""
    try:
        m8 = (mask.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if approx is not None and len(approx) == 4:
            q = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(c)
            q = cv2.boxPoints(rect).astype(np.float32)
        return order_corners(q)
    except Exception:
        return None


class BarSegDetector:
    """Barcode detector using YOLO for detection and SAM2 for segmentation.

    This integrates the approach from AntonAshraf/Barcode-segmentation:
      - YOLO: detects barcode boxes
      - SAM2: segments each detection using a positive point near the box center

    Requirements:
      - ultralytics (YOLO)
      - sam2 package from the external repo on PYTHONPATH, providing
        `sam2.build_sam.build_sam2` and `sam2.sam2_image_predictor.SAM2ImagePredictor`

    Notes:
      - `last_masks` is populated after each detect() call with a list of boolean masks
        aligned to the original image HxW.
    """

    def __init__(
        self,
        weights: str = "models/YOLOV8s_Barcode_Detection.pt",
        sam2_checkpoint: str = "models/sam2.1_hiera_large.pt",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.weights = weights
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.device = _select_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.classes = classes

        # YOLO model
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. `uv pip install ultralytics`.") from e
        try:
            self.yolo = YOLO(self.weights)
        except Exception:
            self.yolo = YOLO("yolov8s.pt")
        try:
            self.yolo.to(self.device)
        except Exception:
            pass

        # SAM2 model (from external repo). We import lazily and raise a clear error if missing.
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "SAM2 not available. Clone AntonAshraf/Barcode-segmentation (or SAM2) so that 'sam2' is importable, "
                "and provide valid sam2_checkpoint + sam2_config."
            ) from e

        if not os.path.exists(self.sam2_checkpoint):
            # Do not crash on init; keep a helpful error for first use
            # Users may pass absolute paths; we only check existence here.
            pass

        try:
            # Pass device explicitly to avoid default 'cuda' inside SAM2 builder
            # Silence SAM2 internal logs to keep progress bars clean
            try:
                lg = logging.getLogger("sam2")
                lg.setLevel(logging.WARNING)
                lg.propagate = False
            except Exception:
                pass
            self._sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(self._sam2_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SAM2 with cfg='{self.sam2_config}', ckpt='{self.sam2_checkpoint}'."
            ) from e

        self.last_masks: List[np.ndarray] = []  # boolean masks recorded per image

    def _yolo_predict(self, img: np.ndarray):
        return self.yolo.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
        )

    def _segment_box(self, image: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        # Use SAM2 point prompting: center of the box, positive label
        try:
            pt = _center_point(box_xyxy)
            labels = np.array([1], dtype=np.int32)
            masks, scores, _ = self._predictor.predict(
                point_coords=pt,
                point_labels=labels,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return None
            # pick top-1 by score
            idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
            m = masks[idx]
            # ensure boolean mask HxWx1 or HxW -> HxW bool
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            return m.astype(bool)
        except Exception:
            return None

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        H, W = image.shape[:2]
        res = self._yolo_predict(image)
        quads: List[np.ndarray] = []
        self.last_masks = []
        if not res:
            return quads
        r0 = res[0]

        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return quads
        # Precompute image embeddings once per image for SAM2
        try:
            self._predictor.set_image(image)
        except Exception:
            pass
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.ones((xyxy.shape[0],), dtype=float)
        except Exception:
            xyxy, confs = None, None
        if xyxy is None:
            return quads

        for (x1, y1, x2, y2, *_), _sc in zip(xyxy, confs):
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            # Segment with SAM2
            mask = self._segment_box(image, box)
            quad: Optional[np.ndarray] = None
            if mask is not None:
                quad = _mask_to_quad(mask)
                self.last_masks.append(mask)
            if quad is None:
                # fallback to the rotated rect on the box itself if segmentation failed
                x1i = int(max(0, min(W - 1, x1)))
                y1i = int(max(0, min(H - 1, y1)))
                x2i = int(max(0, min(W - 1, x2)))
                y2i = int(max(0, min(H - 1, y2)))
                q = np.array([[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]], dtype=np.float32)
                quad = order_corners(q)
            # clip to image
            quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
            quads.append(quad.astype(np.float32))
        return quads
