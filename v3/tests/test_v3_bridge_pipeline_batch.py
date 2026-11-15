import json
import os

import cv2
import numpy as np
import pytest

from edge_drone_pipeline.batch_runner import build_pdf, iter_inputs, load_findings, run_batch
from edge_drone_pipeline.pipeline import EdgeDronePipeline, load_config
from edge_drone_pipeline.sahi_rtdetr_engine import Det, FunctionDetector
from edge_drone_pipeline.vlm_arbitration_bridge import (
    ArbitrationBridge,
    ArbitrationRequest,
    OfflineQueue,
    build_prompt,
    build_request,
)


def dark_box_detector(tile):
    mask = tile.max(axis=2) < 80
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return []
    return [Det(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 0.9, "crack")]


def cfg_for_tests():
    cfg = load_config()
    cfg["conditioner"]["blur_laplacian_min"] = 0.0     # synthetic frames are nearly flat
    cfg["conditioner"]["apply_clahe"] = False
    cfg["sahi"]["full_frame_pass"] = False
    return cfg


def crack_frame(w=1920, h=1080, thickness=9, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(150 + rng.normal(0, 2, (h, w, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
    cv2.line(img, (600, 300), (1200, 600), (30, 30, 30), thickness)
    return img


def make_pipeline(**kw):
    return EdgeDronePipeline(FunctionDetector(dark_box_detector), cfg=cfg_for_tests(), **kw)


# ---------------------------------------------------------------- bridge
def test_request_roundtrip_and_prompt():
    crop = np.zeros((50, 60, 3), np.uint8)
    req = build_request(1, 10, 2.0, crop, np.zeros((400, 800, 3), np.uint8), 1.2, True, {"codebrim": 0.7})
    again = ArbitrationRequest.from_json(req.to_json())
    assert again.track_id == 1 and again.crack_width_mm == 1.2
    prompt = build_prompt(again)
    assert "1.20 mm" in prompt and "structural safety verdict" in prompt


def test_offline_queue_holds_requests_until_connectivity_returns():
    online = {"up": False}
    sent = []

    def sender(req):
        if not online["up"]:
            raise ConnectionError("no signal in tunnel")
        sent.append(req.track_id)
        return {"is_defect": True}

    bridge = ArbitrationBridge(sender, OfflineQueue())
    req = build_request(7, 1, 0.5, np.zeros((20, 20, 3), np.uint8), np.zeros((40, 40, 3), np.uint8))
    assert bridge.submit(req) is None
    assert bridge.queue.counts() == {"pending": 1}
    assert bridge.flush()["still_offline"] is True and sent == []
    online["up"] = True
    out = bridge.flush()
    assert out["sent"] == 1 and sent == [7] and bridge.queue.counts() == {"done": 1}


def test_non_network_errors_are_not_swallowed():
    def sender(req):
        raise ValueError("bad payload")

    bridge = ArbitrationBridge(sender)
    with pytest.raises(ValueError):
        bridge.submit(build_request(1, 1, 0, np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)))


# -------------------------------------------------------------- pipeline
def test_image_finding_has_width_severity_and_timings():
    pipe = make_pipeline(gsd_override_mm_per_px=0.3)
    res = pipe.process_image(crack_frame())
    assert res.accepted and len(res.findings) == 1
    f = res.findings[0]
    assert f["defect_class"] == "crack" and f["gsd_basis"] == "reference"
    assert 2.0 <= f["crack_width_mm"] <= 3.3 and f["width_resolvable"]
    assert f["recommendation"]["severity"] == "urgent"           # >= 2 mm band
    assert f["recommendation"]["needs_engineer_review"] is True
    assert f["min_measurable_width_mm"] == pytest.approx(0.6)
    assert {"condition", "sahi", "zoom", "analyzer", "metrology", "total"} <= set(res.timings_ms)
    assert f["source"] == "stub_not_evaluated"                   # no invented model scores


def test_distance_based_gsd_makes_a_far_thin_crack_an_upper_bound():
    pipe = make_pipeline()                                        # default config: 10 m, ~3.9 mm/px
    f = pipe.process_image(crack_frame(thickness=1)).findings
    if f:                                                         # a 1-px line may fall below the detector
        assert f[0]["width_resolvable"] is False
        assert "provisional" in f[0]["recommendation"]["severity_basis"]


def test_bad_frame_is_rejected_with_a_reason():
    cfg = cfg_for_tests()
    cfg["conditioner"]["clip_low_max_frac"] = 0.2
    pipe = EdgeDronePipeline(FunctionDetector(dark_box_detector), cfg=cfg)
    res = pipe.process_image(np.zeros((1080, 1920, 3), np.uint8))
    assert not res.accepted and any("crushed" in r for r in res.reasons) and res.findings == []


def test_video_triggers_once_after_three_consecutive_frames(tmp_path):
    path = str(tmp_path / "pass.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1920, 1080))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    frame = crack_frame()
    for _ in range(8):
        writer.write(frame)
    writer.release()
    pipe = make_pipeline(gsd_override_mm_per_px=0.3)
    results = list(pipe.process_video(path, stride_s=0.1))
    triggered = [r.index for r in results if r.findings]
    assert len(results) == 8 and len(triggered) == 1 and triggered[0] == 2   # third sampled frame


def test_vlm_bridge_is_used_and_offline_requests_are_marked():
    def offline(req):
        raise ConnectionError("tunnel")

    pipe = make_pipeline(gsd_override_mm_per_px=0.3, bridge=ArbitrationBridge(offline))
    f = pipe.process_image(crack_frame()).findings[0]
    assert f["vlm"] == "queued_offline" and pipe.bridge.queue.counts() == {"pending": 1}


# ----------------------------------------------------------------- batch
def test_batch_resume_error_isolation_and_pdf(tmp_path):
    data = tmp_path / "survey"
    data.mkdir()
    cv2.imwrite(str(data / "a_crack.png"), crack_frame(seed=1))
    cv2.imwrite(str(data / "b_crack.png"), crack_frame(thickness=5, seed=2))
    cv2.imwrite(str(data / "c_black.png"), np.zeros((1080, 1920, 3), np.uint8))
    (data / "d_corrupt.jpg").write_bytes(b"not an image")
    assert len(iter_inputs([str(data)])) == 4

    out = str(tmp_path / "out")
    summary = run_batch([str(data)], make_pipeline(gsd_override_mm_per_px=0.3), out)
    assert summary["total"] == 4 and summary["errors"] == 1 and summary["rejected_quality"] == 1
    assert summary["processed"] == 3
    findings = load_findings(out)
    assert len(findings) == 2 and all("image_path" in f for f in findings)
    assert all(os.path.exists(f["image_path"]) for f in findings)

    again = run_batch([str(data)], make_pipeline(gsd_override_mm_per_px=0.3), out)
    assert again["skipped_resume"] == 4 and again["processed"] == 0
    assert len(open(os.path.join(out, "results.jsonl")).read().strip().splitlines()) == 4

    pdf = build_pdf(out, {"site": "Test bridge"}, max_detail=1)
    data_bytes = open(pdf, "rb").read()
    assert data_bytes.startswith(b"%PDF") and len(data_bytes) > 3000


def test_interrupted_last_line_is_tolerated(tmp_path):
    out = tmp_path / "out"
    (out / "crops").mkdir(parents=True)
    (out / "results.jsonl").write_text(
        json.dumps({"source": "x.png", "status": "ok", "findings": []}) + "\n" + '{"source": "y.p')
    summary = run_batch([], make_pipeline(), str(out))
    assert summary["total"] == 0
    assert load_findings(str(out)) == []
