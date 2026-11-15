"""sahi_rtdetr_engine.py - SAHI-style sliced inference with batched detection.

SAHI (Slicing Aided Hyper Inference): cut a 4K frame into overlapping 640x640
tiles, detect on every tile (batched), shift the boxes back into full-frame
coordinates and merge duplicates from the overlaps. This keeps small cracks at
native resolution instead of shrinking the whole frame to the detector's input.

The detector is pluggable (``detect_batch``). ``RTDETRDetector`` wraps
ultralytics RT-DETR (a .pt, .onnx or TensorRT .engine file); it needs weights
trained on infrastructure defects and is not exercised by the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Protocol, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "defect"

    @property
    def xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Detector(Protocol):
    def detect_batch(self, tiles: List[np.ndarray]) -> List[List[Det]]: ...


def tile_starts(length: int, tile: int, overlap: float) -> List[int]:
    """Start offsets along one axis; the last tile is flush with the edge."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def generate_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    tiles = []
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            tiles.append((x, y, min(x + tile, width), min(y + tile, height)))
    return tiles


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: Det, b: Det) -> float:
    inter = _inter(a.xyxy, b.xyxy)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def ios(a: Det, b: Det) -> float:
    """Intersection over the smaller box: 1.0 when one box sits inside the other."""
    inter = _inter(a.xyxy, b.xyxy)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def nms(dets: Sequence[Det], iou_thr: float = 0.5) -> List[Det]:
    kept: List[Det] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(d.label != k.label or iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept


def nmm(dets: Sequence[Det], ios_thr: float = 0.5) -> List[Det]:
    """Non-maximum MERGING: overlapping same-label boxes become their union.

    A crack cut by a tile border shows up as two half-boxes; NMS would keep one
    half, NMM joins them. The merged score is the maximum of its members.
    """
    current = list(dets)
    while True:
        clusters: List[Det] = []
        for d in sorted(current, key=lambda d: d.score, reverse=True):
            for i, c in enumerate(clusters):
                if c.label == d.label and ios(c, d) >= ios_thr:
                    clusters[i] = Det(
                        min(c.x1, d.x1), min(c.y1, d.y1), max(c.x2, d.x2), max(c.y2, d.y2),
                        max(c.score, d.score), c.label,
                    )
                    break
            else:
                clusters.append(d)
        # A merged box can now overlap clusters it was not compared with: repeat to a fixed point.
        if len(clusters) == len(current):
            return clusters
        current = clusters


class FunctionDetector:
    """Adapt ``fn(tile_bgr) -> list[Det]`` (tile coordinates) to the batch API."""

    def __init__(self, fn: Callable[[np.ndarray], List[Det]]):
        self.fn = fn

    def detect_batch(self, tiles: List[np.ndarray]) -> List[List[Det]]:
        return [self.fn(t) for t in tiles]


class RTDETRDetector:  # pragma: no cover - needs ultralytics + trained weights
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        try:
            from ultralytics import RTDETR
        except ImportError as exc:
            raise ImportError("Install ultralytics to use RTDETRDetector.") from exc
        self.model = RTDETR(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect_batch(self, tiles: List[np.ndarray]) -> List[List[Det]]:
        results = self.model.predict(tiles, conf=self.conf, imgsz=self.imgsz, verbose=False)
        out: List[List[Det]] = []
        for r in results:
            dets = []
            for xyxy, score, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy()):
                dets.append(Det(*(float(v) for v in xyxy), float(score), str(r.names[int(cls)])))
            out.append(dets)
        return out


class SahiEngine:
    def __init__(
        self,
        detector: Detector,
        tile: int = 640,
        overlap: float = 0.2,
        conf: float = 0.25,
        merge_ios: float = 0.5,
        batch_size: int = 8,
        full_frame_pass: bool = True,
    ) -> None:
        self.detector = detector
        self.tile = tile
        self.overlap = overlap
        self.conf = conf
        self.merge_ios = merge_ios
        self.batch_size = max(1, batch_size)
        self.full_frame_pass = full_frame_pass

    @classmethod
    def from_config(cls, detector: Detector, cfg: dict) -> "SahiEngine":
        return cls(detector, **(cfg or {}))

    def run(self, frame_bgr: np.ndarray) -> List[Det]:
        h, w = frame_bgr.shape[:2]
        windows = generate_tiles(w, h, self.tile, self.overlap)
        raw: List[Det] = []
        for start in range(0, len(windows), self.batch_size):
            chunk = windows[start : start + self.batch_size]
            tiles = [frame_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in chunk]
            for (x1, y1, _, _), dets in zip(chunk, self.detector.detect_batch(tiles)):
                for d in dets:
                    if d.score >= self.conf:
                        raw.append(replace(d, x1=d.x1 + x1, y1=d.y1 + y1, x2=d.x2 + x1, y2=d.y2 + y1))
        if self.full_frame_pass and max(h, w) > self.tile:
            scale = self.tile / max(h, w)
            small = cv2.resize(frame_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
            for d in self.detector.detect_batch([small])[0]:
                if d.score >= self.conf:
                    raw.append(Det(d.x1 / scale, d.y1 / scale, d.x2 / scale, d.y2 / scale, d.score, d.label))
        merged = nmm(raw, self.merge_ios)
        return [
            Det(max(0.0, d.x1), max(0.0, d.y1), min(float(w), d.x2), min(float(h), d.y2), d.score, d.label)
            for d in merged
        ]
