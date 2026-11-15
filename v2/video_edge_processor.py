"""video_edge_processor.py - sample video, localize, auto-zoom crop, analyze (RODIC V2).

Flow per sampled frame:
    frame -> localizer -> boxes -> padded crop (15-20 %) -> analyzer -> DefectEvent

Frames are sampled at a configurable interval (default 1 per 2 s), so the
analyzer (the heavy part) only sees a few frames. Works on files (mp4 / avi),
webcams (integer index) and RTSP URLs.

The "zoom" here is a digital crop of the original frame. Optical zoom / gimbal
control belongs to the later drone pipeline (V3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Union

import cv2
import numpy as np

from localizer import Box, Localizer

Source = Union[str, int]


@dataclass
class FrameSample:
    index: int
    t_sec: float
    frame: np.ndarray


@dataclass
class DefectEvent:
    frame_index: int
    t_sec: float
    box: tuple              # original-frame box (x1, y1, x2, y2)
    padded_box: tuple       # box after padding, what was cropped
    score: float            # localizer score (heuristic, not a probability)
    crop: np.ndarray        # BGR crop passed to the analyzer
    analysis: dict = field(default_factory=dict)
    repeat: bool = False    # same region as the previous sampled frame
    annotated: Optional[np.ndarray] = None


def is_live_source(source: Source) -> bool:
    if isinstance(source, int):
        return True
    text = str(source)
    return text.isdigit() or text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def open_capture(source: Source) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source!r}")
    return cap


def sample_frames(
    source: Source,
    interval_s: float = 2.0,
    max_samples: Optional[int] = None,
) -> Iterator[FrameSample]:
    """Yield one frame every ``interval_s`` seconds.

    Files: uses the video clock (frame index / fps) and skips the frames in
    between with ``grab()`` so they are never decoded to full images.
    Live sources: uses the wall clock.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be positive.")
    cap = open_capture(source)
    live = is_live_source(source)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not live and fps <= 0:
            fps = 25.0  # assumption when the container reports no fps
        step = max(1, int(round(fps * interval_s))) if not live else 1
        index = -1
        taken = 0
        next_time = 0.0
        start = time.monotonic()
        while True:
            if not cap.grab():
                break
            index += 1
            if live:
                now = time.monotonic() - start
                if now < next_time:
                    continue
                next_time = now + interval_s
                t_sec = now
            else:
                if index % step != 0:
                    continue
                t_sec = index / fps
            ok, frame = cap.retrieve()
            if not ok:
                continue
            yield FrameSample(index, float(t_sec), frame)
            taken += 1
            if max_samples is not None and taken >= max_samples:
                break
    finally:
        cap.release()


def pad_box(box: Sequence[int], width: int, height: int, pad_frac: float = 0.2, min_side: int = 32) -> tuple:
    """Grow a box by ``pad_frac`` of its size on every side, clipped to the
    image, and never smaller than ``min_side`` pixels."""
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(round(bw * pad_frac)), int(round(bh * pad_frac))
    nx1, ny1, nx2, ny2 = x1 - px, y1 - py, x2 + px, y2 + py
    # enforce a minimum crop size, growing evenly around the centre
    if nx2 - nx1 < min_side:
        grow = (min_side - (nx2 - nx1) + 1) // 2
        nx1, nx2 = nx1 - grow, nx2 + grow
    if ny2 - ny1 < min_side:
        grow = (min_side - (ny2 - ny1) + 1) // 2
        ny1, ny2 = ny1 - grow, ny2 + grow
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(width, nx2), min(height, ny2)
    return (nx1, ny1, nx2, ny2)


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_events(frame: np.ndarray, events: Iterable[DefectEvent]) -> np.ndarray:
    out = frame.copy()
    for ev in events:
        x1, y1, x2, y2 = ev.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        px1, py1, px2, py2 = ev.padded_box
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 200, 255), 1)
        label = ev.analysis.get("defect_class") or "anomaly"
        cv2.putText(out, str(label), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def process_samples(
    samples: Iterable[FrameSample],
    localizer: Localizer,
    analyzer: Callable[[np.ndarray], dict],
    pad_frac: float = 0.2,
    min_score: float = 0.0,
    repeat_iou: float = 0.5,
    on_event: Optional[Callable[[DefectEvent], None]] = None,
) -> List[DefectEvent]:
    """Run localize -> padded crop -> analyze over sampled frames.

    A box that overlaps (IoU >= ``repeat_iou``) a box from the previous sampled
    frame is marked ``repeat`` and is not re-analyzed, which saves compute while
    the camera dwells on the same defect.
    """
    events: List[DefectEvent] = []
    previous_boxes: List[tuple] = []
    for sample in samples:
        height, width = sample.frame.shape[:2]
        current_boxes: List[tuple] = []
        frame_events: List[DefectEvent] = []
        for box in localizer.localize(sample.frame):
            if box.score < min_score:
                continue
            padded = pad_box(box.xyxy, width, height, pad_frac)
            crop = sample.frame[padded[1] : padded[3], padded[0] : padded[2]].copy()
            if crop.size == 0:
                continue
            is_repeat = any(iou(box.xyxy, prev) >= repeat_iou for prev in previous_boxes)
            analysis = {} if is_repeat else analyzer(crop)
            event = DefectEvent(
                sample.index, sample.t_sec, box.xyxy, padded, box.score, crop, analysis, is_repeat
            )
            frame_events.append(event)
            current_boxes.append(box.xyxy)
        if frame_events:
            annotated = draw_events(sample.frame, frame_events)
            for event in frame_events:
                event.annotated = annotated
                events.append(event)
                if on_event:
                    on_event(event)
        previous_boxes = current_boxes
    return events


def process_video(
    source: Source,
    localizer: Localizer,
    analyzer: Callable[[np.ndarray], dict],
    interval_s: float = 2.0,
    pad_frac: float = 0.2,
    max_samples: Optional[int] = None,
    min_score: float = 0.0,
    on_event: Optional[Callable[[DefectEvent], None]] = None,
) -> List[DefectEvent]:
    return process_samples(
        sample_frames(source, interval_s, max_samples),
        localizer,
        analyzer,
        pad_frac=pad_frac,
        min_score=min_score,
        on_event=on_event,
    )
