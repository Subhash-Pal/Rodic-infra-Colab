"""pipeline.py - wires the five V3 modules into one frame-by-frame pipeline.

    frame_conditioner -> sahi engine -> tracker (video only) -> zoom ->
    metrology -> analyzer (V2 contract) -> recommender -> optional VLM bridge

Per-stage wall-clock timings are recorded in every ``FrameResult``. They are
measurements of the machine that ran the code, not a claim about Jetson Orin.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, List, Optional

import cv2
import numpy as np
import yaml

from . import _v2
from .active_zoom_tracker import (
    GimbalCommand,
    GimbalInterface,
    SimpleIoUTracker,
    pixel_offset_to_gimbal,
    roi_crop,
)
from .frame_conditioner import FrameConditioner
from .sahi_rtdetr_engine import Det, Detector, SahiEngine
from .sensorless_metrology import (
    DistanceEstimator,
    gsd_from_distance,
    measure_from_crop,
    min_measurable_width_mm,
)
from .vlm_arbitration_bridge import ArbitrationBridge, build_request

recommend = _v2.irc_recommender.recommend
normalise_class = _v2.irc_recommender.normalise_class
stub_analyzer = _v2.pipeline_adapter.stub_analyzer

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class FrameResult:
    index: int
    t_sec: float
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    detections: List[Det] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    gimbal_commands: List[GimbalCommand] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


class EdgeDronePipeline:
    def __init__(
        self,
        detector: Detector,
        analyzer: Optional[Callable[[np.ndarray], dict]] = None,
        cfg: Optional[dict] = None,
        gimbal: Optional[GimbalInterface] = None,
        distance: Optional[DistanceEstimator] = None,
        bridge: Optional[ArbitrationBridge] = None,
        gsd_override_mm_per_px: Optional[float] = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.conditioner = FrameConditioner.from_config(self.cfg["conditioner"])
        self.engine = SahiEngine.from_config(detector, self.cfg["sahi"])
        self.tracker = SimpleIoUTracker.from_config(self.cfg["tracker"])
        self.analyzer = analyzer or stub_analyzer
        self.gimbal = gimbal
        self.distance = distance
        self.bridge = bridge
        self.gsd_override = gsd_override_mm_per_px
        self._finding_id = 0

    # ------------------------------------------------------------------ scale
    def _gsd(self, frame: np.ndarray, box) -> tuple:
        m = self.cfg["metrology"]
        if self.gsd_override is not None:
            return self.gsd_override, "reference"
        d = self.distance.distance_m(frame, box) if self.distance else None
        basis = "distance_estimator" if d else "configured_distance"
        d = d or m["distance_m"]
        return gsd_from_distance(d, m["pixel_pitch_um"], m["focal_mm"]), basis

    # --------------------------------------------------------------- findings
    def _finding(self, frame: np.ndarray, box, label: str, det_score: float, index: int, t_sec: float,
                 track_id: Optional[int], result: FrameResult) -> dict:
        z, m = self.cfg["zoom"], self.cfg["metrology"]
        if self.gimbal is not None and z.get("mode", "B").upper() == "A":
            h, w = frame.shape[:2]
            cmd = pixel_offset_to_gimbal(box, w, h, z["hfov_deg"], z["target_frame_fraction"],
                                         z["max_optical_zoom"], z["gimbal_deadband_deg"])
            self.gimbal.send(cmd)
            result.gimbal_commands.append(cmd)

        t0 = time.perf_counter()
        crop, padded = roi_crop(frame, box, z["padding"], z["unsharp_amount"])
        result.timings_ms["zoom"] = result.timings_ms.get("zoom", 0.0) + (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        analysis = self.analyzer(crop) or {}
        result.timings_ms["analyzer"] = result.timings_ms.get("analyzer", 0.0) + (time.perf_counter() - t0) * 1000

        cls = analysis.get("defect_class") or label
        confidence = analysis.get("confidence")
        if confidence is None:
            confidence = det_score

        t0 = time.perf_counter()
        gsd, gsd_basis = self._gsd(frame, box)
        measurement = None
        if normalise_class(cls) in ("crack", "unclassified"):
            measurement = measure_from_crop(crop, gsd, m["min_resolvable_px"], m["distance_rel_error"])
        result.timings_ms["metrology"] = result.timings_ms.get("metrology", 0.0) + (time.perf_counter() - t0) * 1000

        # Report the measurement whenever one exists; the recommender only uses it for cracks.
        width = measurement.width_mm_for_decision if measurement else None
        resolvable = measurement.resolvable if measurement else True
        rec = recommend(cls, confidence, width, width_resolvable=resolvable)

        self._finding_id += 1
        finding = {
            "id": self._finding_id, "track_id": track_id, "frame_index": index, "t_sec": t_sec,
            "box": tuple(int(round(v)) for v in box), "padded_box": padded, "image": crop,
            "defect_class": cls, "confidence": confidence, "detector_score": det_score,
            "crack_width_mm": width, "width_resolvable": resolvable,
            "gsd_mm_per_px": gsd, "gsd_basis": gsd_basis,
            "min_measurable_width_mm": min_measurable_width_mm(gsd, m["min_resolvable_px"]),
            "measurement": asdict(measurement) if measurement else None,
            "source": analysis.get("source", "n/a"), "recommendation": rec.to_dict(),
        }
        if self.bridge is not None:
            req = build_request(track_id or finding["id"], index, t_sec, crop, frame, width, resolvable,
                                {"detector": det_score, **(analysis.get("scores") or {})})
            response = self.bridge.submit(req)
            finding["vlm"] = response if response is not None else "queued_offline"
        return finding

    # ------------------------------------------------------------------- main
    def process_frame(self, frame: np.ndarray, index: int = 0, t_sec: float = 0.0,
                      use_tracker: bool = True) -> FrameResult:
        result = FrameResult(index, t_sec, accepted=True)

        t0 = time.perf_counter()
        conditioned, report = self.conditioner.process(frame)
        result.timings_ms["condition"] = (time.perf_counter() - t0) * 1000
        if not report.accepted:
            result.accepted, result.reasons = False, report.reasons
            return result

        t0 = time.perf_counter()
        result.detections = self.engine.run(conditioned)
        result.timings_ms["sahi"] = (time.perf_counter() - t0) * 1000

        if use_tracker:
            t0 = time.perf_counter()
            _, triggered = self.tracker.update(result.detections)
            result.timings_ms["track"] = (time.perf_counter() - t0) * 1000
            for tr in triggered:
                result.findings.append(
                    self._finding(conditioned, tr.box, tr.label, tr.score, index, t_sec, tr.id, result))
        else:
            for det in result.detections:
                result.findings.append(
                    self._finding(conditioned, det.xyxy, det.label, det.score, index, t_sec, None, result))
        result.timings_ms["total"] = sum(v for k, v in result.timings_ms.items() if k != "total")
        return result

    def process_image(self, frame: np.ndarray, name_index: int = 0) -> FrameResult:
        """Single still image: no temporal gating, every detection becomes a finding."""
        return self.process_frame(frame, name_index, 0.0, use_tracker=False)

    def process_video(self, source, stride_s: Optional[float] = None,
                      max_samples: Optional[int] = None) -> Iterator[FrameResult]:
        stride = stride_s if stride_s is not None else self.cfg["video"]["stride_s"]
        self.tracker = SimpleIoUTracker.from_config(self.cfg["tracker"])   # fresh tracks per video
        for sample in _v2.video_edge_processor.sample_frames(source, stride, max_samples):
            yield self.process_frame(sample.frame, sample.index, sample.t_sec, use_tracker=True)
