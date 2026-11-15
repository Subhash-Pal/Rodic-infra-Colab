"""frame_conditioner.py - quality gates and low-light conditioning for one frame.

* Blur gate: variance of the Laplacian, measured on a frame resized to a fixed
  width so the threshold does not depend on the 4K / 1080p source resolution.
* Exposure gate: fraction of crushed-shadow and blown-highlight pixels.
* CLAHE on the LAB L channel (contrast-limited adaptive histogram equalization),
  which lifts contrast in tunnels without shifting colour as much as RGB CLAHE.

All thresholds are starting values (see config.yaml); none is validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


def laplacian_variance(frame: np.ndarray, analysis_width: int = 960) -> float:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w > analysis_width:
        gray = cv2.resize(gray, (analysis_width, int(round(h * analysis_width / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clipping_fractions(frame: np.ndarray) -> Tuple[float, float]:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = gray.size
    return float((gray <= 5).sum()) / total, float((gray >= 250).sum()) / total


def apply_clahe_lab(frame_bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a_chan, b_chan)), cv2.COLOR_LAB2BGR)


@dataclass
class ConditionReport:
    accepted: bool
    blur_score: float
    low_clip: float
    high_clip: float
    clahe_applied: bool
    reasons: List[str] = field(default_factory=list)


class FrameConditioner:
    def __init__(
        self,
        blur_laplacian_min: float = 100.0,
        analysis_width: int = 960,
        clip_low_max_frac: float = 0.20,
        clip_high_max_frac: float = 0.10,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile: int = 8,
    ) -> None:
        self.blur_min = blur_laplacian_min
        self.analysis_width = analysis_width
        self.low_max = clip_low_max_frac
        self.high_max = clip_high_max_frac
        self.apply_clahe = apply_clahe
        self.clip_limit = clahe_clip_limit
        self.tile = clahe_tile

    @classmethod
    def from_config(cls, cfg: dict) -> "FrameConditioner":
        return cls(**(cfg or {}))

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, ConditionReport]:
        blur = laplacian_variance(frame_bgr, self.analysis_width)
        low, high = clipping_fractions(frame_bgr)
        reasons: List[str] = []
        if blur < self.blur_min:
            reasons.append(f"blurry (Laplacian variance {blur:.1f} < {self.blur_min})")
        if low > self.low_max:
            reasons.append(f"crushed shadows ({low:.0%} of pixels near black)")
        if high > self.high_max:
            reasons.append(f"blown highlights ({high:.0%} of pixels near white)")
        if reasons:
            return frame_bgr, ConditionReport(False, blur, low, high, False, reasons)
        out = apply_clahe_lab(frame_bgr, self.clip_limit, self.tile) if self.apply_clahe else frame_bgr
        return out, ConditionReport(True, blur, low, high, self.apply_clahe)
