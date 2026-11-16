"""Smoke tests of the Streamlit script itself (no model download: the stub analyzer is forced)."""

import os

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app_logic import analyze_static, make_pdf_bytes

APP = os.path.join(os.path.dirname(__file__), "..", "app_v2.py")


@pytest.fixture(autouse=True)
def force_stub(monkeypatch):
    monkeypatch.setenv("RODIC_ANALYZER", "pipeline_adapter:stub_analyzer")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RODIC_ENABLE_LIVE_SOURCES", raising=False)


def new_app():
    return AppTest.from_file(APP, default_timeout=60)


def test_first_render_is_clean_and_says_the_analyzer_is_a_stub():
    at = new_app().run()
    assert not at.exception
    assert at.title[0].value.startswith("RODIC InfraAI")
    assert any("stub analyzer" in i.value for i in at.info)
    assert any("Screening aid only" in w.value for w in at.warning)


def test_video_mode_hides_live_sources_unless_enabled_and_renders():
    at = new_app().run()
    at.sidebar.radio[0].set_value("Live Video Stream (Edge Auto-Zoom)").run()
    assert not at.exception
    assert list(source_radio(at).options) == ["Upload mp4 / avi"]


def source_radio(at):
    return next(r for r in at.radio if r.label == "Source")


def test_live_sources_appear_on_an_edge_device(monkeypatch):
    monkeypatch.setenv("RODIC_ENABLE_LIVE_SOURCES", "1")
    at = new_app().run()
    at.sidebar.radio[0].set_value("Live Video Stream (Edge Auto-Zoom)").run()
    assert not at.exception
    assert list(source_radio(at).options) == ["Upload mp4 / avi", "Webcam index", "RTSP URL"]


def test_results_survive_a_rerun_and_render():
    frame = np.full((120, 160, 3), 128, np.uint8)
    findings = analyze_static(frame, lambda c: {
        "defect_class": "Crack", "confidence": 0.8, "source": "fake", "cv_predictions": [{"class": "crack"}],
        "scores": {"codebrim": {"crack": 0.8, "background": 0.1}, "sdnet_crack_probability": 0.4}}, 2.2)
    findings[0]["evidence_state"] = "PARTIAL_AGREEMENT"
    run = {"mode": "Static Image", "findings": findings, "analyzer": "fake",
           "pdf": make_pdf_bytes(findings, {})}
    at = new_app()
    at.session_state["run"] = run
    at.run()
    assert not at.exception
    assert any("Finding 1" in e.label for e in at.expander)
    at.run()                                                              # a second rerun keeps the results
    assert not at.exception and any("Finding 1" in e.label for e in at.expander)
