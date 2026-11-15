"""active_zoom_tracker.py - temporal gating and the two zoom modes.

Tracking: a greedy IoU tracker (a simple stand-in for ByteTrack / OC-SORT). A
zoom is triggered only after a track has been seen in ``confirm_frames``
CONSECUTIVE frames, and only once per track, so single-frame false positives
never move the gimbal.

Zoom Mode A (gimbal / optical): convert the target's pixel offset from the image
centre into yaw / pitch with the pinhole model, plus a zoom factor. This module
only computes the command and hands it to a ``GimbalInterface``. The shipped
``LoggingGimbal`` records commands; NO MAVLink or vendor-SDK control is
implemented here, so nothing in this repo moves real hardware.

Zoom Mode B (lossless crop): crop the padded ROI from the native 4K frame and
sharpen it with an unsharp mask. A learned super-resolution step (ESRGAN-Lite)
can be passed in as ``enhance``; none is bundled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import _v2
from .sahi_rtdetr_engine import Det, iou

pad_box = _v2.video_edge_processor.pad_box


@dataclass
class Track:
    id: int
    box: tuple
    label: str
    score: float
    consecutive: int = 1
    missed: int = 0
    hits: int = 1
    fired: bool = False


class SimpleIoUTracker:
    def __init__(self, iou_match: float = 0.3, confirm_frames: int = 3, max_age: int = 5) -> None:
        self.iou_match = iou_match
        self.confirm_frames = confirm_frames
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    @classmethod
    def from_config(cls, cfg: dict) -> "SimpleIoUTracker":
        return cls(**(cfg or {}))

    def update(self, dets: Sequence[Det]) -> Tuple[List[Track], List[Track]]:
        """Return (live tracks, tracks that became confirmed on this frame)."""
        pairs = []
        for tid, tr in self.tracks.items():
            for di, d in enumerate(dets):
                if d.label != tr.label:
                    continue
                overlap = iou(Det(*tr.box, tr.score, tr.label), d)
                if overlap >= self.iou_match:
                    pairs.append((overlap, tid, di))
        pairs.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            tr, d = self.tracks[tid], dets[di]
            tr.box, tr.score = d.xyxy, d.score
            tr.consecutive += 1
            tr.hits += 1
            tr.missed = 0
            used_tracks.add(tid)
            used_dets.add(di)

        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks:
                tr.missed += 1
                tr.consecutive = 0
                if tr.missed > self.max_age:
                    del self.tracks[tid]

        for di, d in enumerate(dets):
            if di not in used_dets:
                self.tracks[self._next_id] = Track(self._next_id, d.xyxy, d.label, d.score)
                self._next_id += 1

        triggered = []
        for tr in self.tracks.values():
            if not tr.fired and tr.consecutive >= self.confirm_frames:
                tr.fired = True
                triggered.append(tr)
        return list(self.tracks.values()), triggered


@dataclass
class GimbalCommand:
    yaw_deg: float
    pitch_deg: float
    zoom_factor: float


def pixel_offset_to_gimbal(
    box: Sequence[float],
    frame_w: int,
    frame_h: int,
    hfov_deg: float,
    target_frame_fraction: float = 0.5,
    max_zoom: float = 5.0,
    deadband_deg: float = 1.0,
) -> GimbalCommand:
    """Pinhole model: focal length in pixels f = (W/2) / tan(HFOV/2).

    yaw   = atan(dx / f)   with dx = box centre x - W/2  (right positive)
    pitch = -atan(dy / f)  with dy = box centre y - H/2  (image y points down,
                           so a target above centre gives a positive pitch / up)
    Square pixels are assumed, so the same f is used on both axes.
    The zoom factor makes the box fill ``target_frame_fraction`` of the limiting
    frame dimension, clamped to [1, max_zoom].
    """
    f_px = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    yaw = math.degrees(math.atan((cx - frame_w / 2.0) / f_px))
    pitch = -math.degrees(math.atan((cy - frame_h / 2.0) / f_px))
    if abs(yaw) < deadband_deg:
        yaw = 0.0
    if abs(pitch) < deadband_deg:
        pitch = 0.0
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    zoom = target_frame_fraction * min(frame_w / bw, frame_h / bh)
    return GimbalCommand(yaw, pitch, float(min(max(zoom, 1.0), max_zoom)))


class GimbalInterface:
    def send(self, command: GimbalCommand) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LoggingGimbal(GimbalInterface):
    """Records commands only. Replace with a MAVLink / SDK adapter on the vehicle."""

    def __init__(self) -> None:
        self.sent: List[GimbalCommand] = []

    def send(self, command: GimbalCommand) -> None:
        self.sent.append(command)


def unsharp(img: np.ndarray, amount: float = 0.6, sigma: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def roi_crop(
    frame: np.ndarray,
    box: Sequence[float],
    padding: float = 0.25,
    unsharp_amount: float = 0.6,
    enhance: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, tuple]:
    """Mode B: lossless padded ROI crop from the native frame, then sharpen."""
    h, w = frame.shape[:2]
    padded = pad_box(tuple(int(round(v)) for v in box), w, h, padding)
    crop = frame[padded[1] : padded[3], padded[0] : padded[2]].copy()
    if enhance is not None:
        crop = enhance(crop)
    if unsharp_amount > 0:
        crop = unsharp(crop, unsharp_amount)
    return crop, padded
