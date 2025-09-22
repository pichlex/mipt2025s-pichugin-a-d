from __future__ import annotations

from typing import Protocol, List, Any

import numpy as np


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Return list of quads (each 4x2 float32)."""
        ...


def get_detector(name: str = "auto", **kwargs: Any) -> Detector:
    n = (name or "auto").lower()
    if n in ("auto", "combined", "default"):
        from .auto import AutoDetector

        return AutoDetector(**kwargs)
    if n in ("yolo", "ultralytics"):
        from .yolo import YOLODetector

        return YOLODetector(**kwargs)
    if n in ("barseg", "sam2", "yolo+sam2", "barcode-segmentation"):
        # YOLO for detection + SAM2 for segmentation (from external repo)
        from .barseg import BarSegDetector

        return BarSegDetector(**kwargs)
    if n in ("opencv", "cv2"):
        from .auto import AutoDetector

        return AutoDetector(**kwargs)
    raise ValueError(f"Unknown detector '{name}'. Options: auto | yolo | barseg")
