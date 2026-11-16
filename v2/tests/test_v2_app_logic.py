import os
import sys

import cv2
import numpy as np
import pytest

from app_logic import (
    MAX_GEMINI_CALLS,
    analyze_static,
    analyze_video,
    attach_gemini,
    decode_image_bytes,
    make_pdf_bytes,
    thumbnail,
)
from localizer import OpenCVAnomalyLocalizer

HOSTED = os.path.join(os.path.dirname(__file__), "..", "hosted")


def png_bytes(h=120, w=160):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 128, np.uint8))
    return buf.tobytes()


def fake_analyzer(cls="Crack", conf=0.9):
    return lambda crop: {"defect_class": cls, "confidence": conf, "source": "fake",
                         "scores": {"codebrim": {"crack": conf}, "sdnet_crack_probability": 0.2},
                         "cv_predictions": [{"class": cls.lower()}] if cls else []}


def test_image_validation_matches_hosted_rules():
    assert decode_image_bytes(png_bytes()).shape[:2] == (120, 160)
    for bad in (None, b"", b"not an image"):
        with pytest.raises(ValueError):
            decode_image_bytes(bad)
    with pytest.raises(ValueError, match="10 MB"):
        decode_image_bytes(b"0" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="too small"):
        decode_image_bytes(png_bytes(20, 200))


def test_static_finding_uses_width_for_severity():
    frame = np.full((200, 300, 3), 120, np.uint8)
    f = analyze_static(frame, fake_analyzer("Crack"), width_mm=2.5)[0]
    assert f["recommendation"]["severity"] == "urgent" and f["box"] == (0, 0, 300, 200)
    unknown = analyze_static(frame, fake_analyzer("Crack"), width_mm=None)[0]
    assert "provisional" in unknown["recommendation"]["severity_basis"]
    stub_like = analyze_static(frame, lambda c: {"defect_class": None, "confidence": None, "source": "stub"})[0]
    assert stub_like["recommendation"]["severity"] == "watch"          # unclassified -> human inspection


def test_video_run_returns_findings_and_reports_progress(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    frame = np.full((240, 320, 3), 170, np.uint8)
    cv2.line(frame, (40, 60), (280, 180), (40, 40, 40), 3)
    for _ in range(40):
        writer.write(frame)
    writer.release()
    ticks = []
    findings = analyze_video(path, OpenCVAnomalyLocalizer(), fake_analyzer("Spallation"), interval_s=1.0,
                             max_samples=4, on_sample=lambda n, m: ticks.append((n, m)))
    assert ticks == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert findings and findings[0]["defect_class"] == "Spallation"
    assert findings[0]["overview"].shape[1] <= 960                      # thumbnails, not full frames
    assert any(f["repeat"] for f in findings[1:])


def test_thumbnail_only_shrinks():
    assert thumbnail(np.zeros((2000, 4000, 3), np.uint8), 960).shape[:2] == (480, 960)
    assert thumbnail(np.zeros((100, 200, 3), np.uint8), 960).shape[:2] == (100, 200)


def make_findings(n, severity_widths=None):
    frame = np.full((100, 100, 3), 120, np.uint8)
    out = []
    for i in range(n):
        f = analyze_static(frame, fake_analyzer("Crack"), width_mm=(severity_widths or [1.0] * n)[i])[0]
        f["id"] = i + 1
        out.append(f)
    return out


def test_gemini_is_capped_and_prefers_the_most_severe_findings():
    findings = make_findings(8, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.0, 3.0])
    calls = []
    done = attach_gemini(findings, lambda b: calls.append(1) or {"observations": []},
                         lambda cv, g: {"arbitration": {"evidence_state": "HIGH_AGREEMENT"}}, max_calls=99)
    assert done == MAX_GEMINI_CALLS == len(calls)
    assert findings[6]["evidence_state"] == "HIGH_AGREEMENT" and findings[7]["evidence_state"] == "HIGH_AGREEMENT"


def test_gemini_failure_is_recorded_and_cv_result_survives():
    findings = make_findings(1)

    def broken(_):
        raise ConnectionError("down")

    attach_gemini(findings, broken, lambda cv, g: {"arbitration": {"evidence_state": "PARTIAL_AGREEMENT"}})
    f = findings[0]
    assert f["gemini_error"] == "ConnectionError" and "gemini" not in f
    assert f["evidence_state"] == "PARTIAL_AGREEMENT" and f["recommendation"]["severity"]   # CV side intact


def test_original_upload_bytes_are_sent_instead_of_the_thumbnail_and_repeats_are_skipped():
    findings = make_findings(2)
    findings[0]["gemini_bytes"] = b"ORIGINAL"
    findings[1]["repeat"] = True
    sent = []
    attach_gemini(findings, lambda b: sent.append(b) or {}, lambda cv, g: {"arbitration": {"evidence_state": "X"}})
    assert sent == [b"ORIGINAL"]


def test_real_hosted_arbitration_consumes_the_analyzer_output():
    sys.path.insert(0, HOSTED)
    from rodic_inference.evidence_arbitration import build_inspection_evidence_record

    findings = make_findings(1)
    gemini = {"observations": [{"evidence_type": "crack", "observation": "x", "location_description": "y",
                                "visual_confidence": "high"}]}
    attach_gemini(findings, lambda b: gemini, build_inspection_evidence_record)
    assert findings[0]["evidence_state"] == "HIGH_AGREEMENT"
    assert findings[0]["arbitration"]["human_validation"]["required"] is True


def test_pdf_bytes_include_gemini_evidence_state_without_error():
    findings = make_findings(2, [0.1, 2.5])
    findings[1]["evidence_state"] = "MODEL_CONFLICT"
    data = make_pdf_bytes(findings, {"site": "x"})
    assert data.startswith(b"%PDF") and len(data) > 3000
