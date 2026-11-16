"""app_logic.py - everything the V2 app does that is not drawing widgets (so it can be tested).

Turns analyzed crops into findings (class, confidence, width, recommendation), optionally attaches Gemini
visual evidence + the hosted evidence arbitration, and builds the PDF bytes.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from irc_recommender import recommend
from localizer import Localizer
from pdf_exporter import export_report
from video_edge_processor import DefectEvent, FrameSample, process_samples, sample_frames

MAX_GEMINI_CALLS = 5          # hard cap so one video run cannot burn the API quota


def thumbnail(img: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def jpeg_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed.")
    return buf.tobytes()


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def decode_image_bytes(data: Optional[bytes]) -> np.ndarray:
    """Same validation rules as the hosted app: non-empty, <= 10 MB, decodable, at least 32 x 32."""
    if not data:
        raise ValueError("No image data supplied.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit.")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV could not decode the uploaded image.")
    if min(frame.shape[:2]) < 32:
        raise ValueError("Image dimensions are too small.")
    return frame


def event_to_finding(index: int, event: DefectEvent, width_mm: Optional[float] = None) -> dict:
    analysis = event.analysis or {}
    rec = recommend(analysis.get("defect_class"), analysis.get("confidence"), width_mm)
    return {
        "id": index, "frame_index": event.frame_index, "t_sec": event.t_sec, "box": event.box,
        "image": thumbnail(event.crop, 1024), "overview": event.annotated, "repeat": event.repeat,
        "defect_class": analysis.get("defect_class"), "confidence": analysis.get("confidence"),
        "crack_width_mm": width_mm, "source": analysis.get("source", "n/a"),
        "analysis": analysis, "recommendation": rec.to_dict(),
    }


def analyze_static(frame_bgr: np.ndarray, analyzer: Callable[[np.ndarray], dict],
                   width_mm: Optional[float] = None) -> List[dict]:
    """Whole-image analysis (the hosted app's flow) as a single finding."""
    h, w = frame_bgr.shape[:2]
    event = DefectEvent(0, 0.0, (0, 0, w, h), (0, 0, w, h), 1.0, frame_bgr, analyzer(frame_bgr))
    return [event_to_finding(1, event, width_mm)]


def analyze_video(
    source,
    localizer: Localizer,
    analyzer: Callable[[np.ndarray], dict],
    interval_s: float = 2.0,
    pad_frac: float = 0.2,
    max_samples: int = 60,
    min_score: float = 0.0,
    width_mm: Optional[float] = None,
    on_sample: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    def samples():
        for n, s in enumerate(sample_frames(source, interval_s, max_samples), start=1):
            if on_sample:
                on_sample(n, max_samples)
            yield s

    events = process_samples(samples(), localizer, analyzer, pad_frac=pad_frac, min_score=min_score)
    return [event_to_finding(i, ev, width_mm) for i, ev in enumerate(events, start=1)]


def attach_gemini(
    findings: List[dict],
    runner: Callable[[bytes], dict],
    record_builder: Callable[[dict, Optional[dict]], dict],
    max_calls: int = MAX_GEMINI_CALLS,
) -> int:
    """Run Gemini on up to ``max_calls`` findings (highest severity first, repeats skipped) and attach the
    hosted evidence arbitration. A Gemini failure is recorded on the finding; CV results stay available."""
    order = ["none", "cosmetic", "watch", "significant", "urgent"]
    candidates = [f for f in findings if not f.get("repeat")]
    candidates.sort(key=lambda f: -order.index(f["recommendation"]["severity"]))
    done = 0
    for f in candidates[: min(max_calls, MAX_GEMINI_CALLS)]:
        cv_result = {"predictions": f["analysis"].get("cv_predictions", [])}
        gemini = None
        try:
            data = f.pop("gemini_bytes", None) or jpeg_bytes(f["image"])   # original upload if provided
            gemini = runner(data)
            f["gemini"] = gemini
        except Exception as exc:
            f["gemini_error"] = type(exc).__name__
        try:
            record = record_builder(cv_result, gemini)
            f["arbitration"] = record["arbitration"]
            f["evidence_state"] = record["arbitration"]["evidence_state"]
        except Exception as exc:
            f["arbitration_error"] = type(exc).__name__
        done += 1
    return done


def make_pdf_bytes(findings: Iterable[dict], meta: Optional[dict] = None, max_detail: int = 50) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        export_report(path, list(findings), meta or {}, max_detail=max_detail)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(path)
