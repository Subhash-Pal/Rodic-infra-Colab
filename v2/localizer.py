"""localizer.py - lightweight CPU defect localizers for RODIC V2.

Two options behind one interface ``localize(frame_bgr) -> list[Box]``:

* ``OpenCVAnomalyLocalizer`` - heuristic fallback (dark thin structures via a
  morphological black-hat). It needs no weights and runs on CPU, but it is NOT
  a trained detector and is NOT validated: expect false positives on shadows,
  joints and stains. Use it to demonstrate the zoom flow, not to claim accuracy.
* ``UltralyticsLocalizer`` - YOLOv8n / RT-DETR through the ``ultralytics``
  package. Generic COCO weights do not know about cracks: you must supply
  weights trained on infrastructure defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    label: str = "anomaly"

    @property
    def xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


class Localizer(Protocol):
    def localize(self, frame_bgr: np.ndarray) -> List[Box]: ...


class OpenCVAnomalyLocalizer:
    """Find dark, elongated structures (crack-like) with black-hat filtering."""

    def __init__(
        self,
        kernel: int = 15,
        min_contrast: float = 25.0,
        min_area: int = 80,
        min_aspect: float = 3.0,
        max_boxes: int = 5,
    ) -> None:
        self.kernel = kernel
        self.min_contrast = min_contrast
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_boxes = max_boxes

    def localize(self, frame_bgr: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel, self.kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        if float(blackhat.max()) < self.min_contrast:
            return []
        _, mask = cv2.threshold(blackhat, self.min_contrast, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Box] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area and len(contour) < 15:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            short, long_ = sorted((rw, rh))
            if long_ < 1 or long_ / max(short, 1.0) < self.min_aspect:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region = blackhat[y : y + h, x : x + w]
            score = float(region.mean()) / 255.0
            boxes.append(Box(x, y, x + w, y + h, score, "anomaly"))
        boxes.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        return boxes[: self.max_boxes]


class UltralyticsLocalizer:
    """YOLOv8n / RT-DETR wrapper. Requires ``pip install ultralytics`` and
    domain-trained weights (``weights`` path)."""

    def __init__(self, weights: str, conf: float = 0.25, rtdetr: bool = False, imgsz: int = 640):
        try:
            from ultralytics import RTDETR, YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install ultralytics to use UltralyticsLocalizer.") from exc
        self.model = RTDETR(weights) if rtdetr else YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def localize(self, frame_bgr: np.ndarray) -> List[Box]:  # pragma: no cover
        result = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = result.names
        boxes: List[Box] = []
        for xyxy, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            boxes.append(Box(x1, y1, x2, y2, float(score), str(names[int(cls)])))
        return boxes


def make_localizer(kind: str = "opencv", weights: Optional[str] = None) -> Localizer:
    if kind == "opencv":
        return OpenCVAnomalyLocalizer()
    if kind in ("yolo", "rtdetr"):
        if not weights:
            raise ValueError("Defect-trained weights are required for yolo / rtdetr.")
        return UltralyticsLocalizer(weights, rtdetr=(kind == "rtdetr"))
    raise ValueError(f"Unknown localizer kind: {kind}")
