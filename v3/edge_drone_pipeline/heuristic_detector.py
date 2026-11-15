"""heuristic_detector.py - weight-free tile detector for demos and smoke tests ONLY.

It wraps V2's OpenCV black-hat localizer so the whole V3 flow can run without a
trained RT-DETR. It is not a validated detector: expect false positives on
shadows, joints and stains. Do not quote its output as model performance.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import _v2
from .sahi_rtdetr_engine import Det


class HeuristicTileDetector:
    def __init__(self) -> None:
        import localizer  # V2 module, on sys.path via _v2

        self._loc = localizer.OpenCVAnomalyLocalizer()

    def detect_batch(self, tiles: List[np.ndarray]) -> List[List[Det]]:
        return [
            [Det(b.x1, b.y1, b.x2, b.y2, min(1.0, 0.5 + b.score), "anomaly") for b in self._loc.localize(t)]
            for t in tiles
        ]
