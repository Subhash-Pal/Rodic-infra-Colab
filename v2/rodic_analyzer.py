"""rodic_analyzer.py - the real CODEBRIM + SDNET2018 analyzer for V2 (hosted-POC models).

Wraps the unchanged hosted inference code in ``v2/hosted`` and returns the analyzer contract used by V2
(see pipeline_adapter.py). Torch and the checkpoints are loaded lazily, so importing this module is cheap and
the rest of V2 still runs (with the stub analyzer) where torch is not installed.

Policy kept from the hosted app: CODEBRIM and SDNET2018 scores are source-specific, uncalibrated, and are
never averaged together.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

HOSTED_DIR = Path(__file__).resolve().parent / "hosted"


def _ensure_hosted_on_path() -> None:
    if str(HOSTED_DIR) not in sys.path:
        sys.path.insert(0, str(HOSTED_DIR))


def classify_scores(codebrim: list, sdnet: dict, threshold: float) -> dict:
    """Pure decision rule (no torch). ``codebrim`` is the hosted list of {defect, score, detected};
    ``sdnet`` is {crack_probability, ...}. Returns the V2 analyzer dict."""
    defects = [r for r in codebrim if r["defect"] != "background" and float(r["score"]) >= threshold]
    defects.sort(key=lambda r: float(r["score"]), reverse=True)
    crack_p = float(sdnet.get("crack_probability", 0.0))

    if defects:
        top, basis = defects[0], "codebrim"
        defect_class, confidence = top["defect"], float(top["score"])
    elif crack_p >= threshold:
        defect_class, confidence, basis = "crack", crack_p, "sdnet_only"
    else:
        defect_class, confidence, basis = "background", None, "none_above_threshold"

    return {
        "defect_class": defect_class,
        "confidence": confidence,
        "class_basis": basis,
        "scores": {
            "codebrim": {r["defect"]: float(r["score"]) for r in codebrim},
            "sdnet_crack_probability": crack_p,
        },
        "cv_predictions": [{"class": r["defect"]} for r in defects],   # what the hosted arbitration consumes
        "threshold": threshold,
        "source": "codebrim+sdnet2018 (uncalibrated, sources not averaged)",
    }


class HostedPipeline:
    """Loads both checkpoints once; ``analyze(crop_bgr, threshold)`` follows the analyzer contract."""

    def __init__(self, ensure: Optional[Callable[[], object]] = None) -> None:
        if ensure is None:
            from deploy_assets import ensure_models as ensure
        ensure()
        _ensure_hosted_on_path()
        from rodic_inference import inference as hosted      # imports torch

        self._hosted = hosted
        self.codebrim_model = hosted.load_codebrim_model()
        self.sdnet_model = hosted.load_sdnet_model()

    def analyze(self, crop_bgr: np.ndarray, threshold: float = 0.50) -> dict:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        codebrim = self._hosted.run_codebrim_inference(self.codebrim_model, rgb)
        sdnet = self._hosted.run_sdnet_inference(self.sdnet_model, rgb)
        return classify_scores(codebrim, sdnet, threshold)
