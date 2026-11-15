"""Single source for the V2 modules that V3 reuses (recommender, PDF, video sampling).

V2 is a flat folder of modules; this helper puts it on ``sys.path`` when needed
so V3 never duplicates that code.
"""

from __future__ import annotations

import os
import sys

_V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "v2"))


def ensure_v2_on_path() -> None:
    if not os.path.isdir(_V2_DIR):
        raise ImportError(f"V2 folder not found at {_V2_DIR}; V3 reuses its modules.")
    if _V2_DIR not in sys.path:
        sys.path.append(_V2_DIR)


ensure_v2_on_path()

import irc_recommender  # noqa: E402
import pdf_exporter  # noqa: E402
import pipeline_adapter  # noqa: E402
import video_edge_processor  # noqa: E402

__all__ = ["irc_recommender", "pdf_exporter", "pipeline_adapter", "video_edge_processor"]
