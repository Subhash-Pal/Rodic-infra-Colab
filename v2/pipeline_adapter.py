"""pipeline_adapter.py - the plug point to the existing RODIC pipeline.

The CODEBRIM + SDNET2018 + Gemini pipeline lives in the app repositories, not in
this one. V2 therefore talks to it through one small contract:

    analyzer(crop_bgr: np.ndarray) -> dict with keys
        defect_class : str | None      e.g. "Crack", "Spallation"
        confidence   : float | None    0..1 (the model's own, uncalibrated)
        scores       : dict            per-source scores, free-form
        source       : str             where the numbers came from

Wire the real pipeline by pointing the environment variable ``RODIC_ANALYZER``
at a function with that signature, e.g.

    RODIC_ANALYZER=main_demo_adapter:analyze_crop

Without it the app uses ``stub_analyzer``, which reports "not evaluated" and
invents nothing, so no report can ever show made-up model scores.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable, Optional

import numpy as np

Analyzer = Callable[[np.ndarray], dict]


def stub_analyzer(crop_bgr: np.ndarray) -> dict:
    return {
        "defect_class": None,
        "confidence": None,
        "scores": {},
        "source": "stub_not_evaluated",
    }


def load_analyzer(spec: Optional[str] = None) -> Analyzer:
    """Load ``module:function`` from ``spec`` or ``$RODIC_ANALYZER``."""
    spec = spec or os.environ.get("RODIC_ANALYZER")
    if not spec:
        return stub_analyzer
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ValueError("Analyzer spec must look like 'module:function'.")
    func = getattr(importlib.import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"{spec} is not callable.")
    return func


def is_stub(analyzer: Analyzer) -> bool:
    return analyzer is stub_analyzer
