import os

import cv2
import numpy as np
import pytest

from irc_recommender import recommend
from localizer import Box, OpenCVAnomalyLocalizer
from pdf_exporter import export_report
from pipeline_adapter import is_stub, load_analyzer, stub_analyzer
from video_edge_processor import iou, pad_box, process_samples, process_video, FrameSample, sample_frames


def crack_frame(h=240, w=320):
    img = np.full((h, w, 3), 170, np.uint8)
    cv2.line(img, (40, 60), (280, 180), (40, 40, 40), 3)
    return img


class FixedLocalizer:
    def __init__(self, boxes):
        self.boxes = boxes

    def localize(self, frame):
        return list(self.boxes)


def test_pad_box_clips_and_pads():
    x1, y1, x2, y2 = pad_box((100, 100, 200, 160), 640, 480, 0.2)
    assert (x1, y1, x2, y2) == (80, 88, 220, 172)
    assert pad_box((0, 0, 50, 50), 100, 100, 0.2)[:2] == (0, 0)          # clipped at 0
    assert pad_box((90, 90, 100, 100), 100, 100, 0.2)[2:] == (100, 100)  # clipped at edge


def test_pad_box_minimum_size():
    x1, y1, x2, y2 = pad_box((100, 100, 104, 104), 640, 480, 0.2, min_side=32)
    assert x2 - x1 >= 32 and y2 - y1 >= 32


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.0 < iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_opencv_localizer_finds_dark_line_and_ignores_plain_frame():
    boxes = OpenCVAnomalyLocalizer().localize(crack_frame())
    assert len(boxes) >= 1
    b = boxes[0]
    assert b.x1 < 120 and b.x2 > 200 and b.y1 < 100 and b.y2 > 140
    assert OpenCVAnomalyLocalizer().localize(np.full((240, 320, 3), 170, np.uint8)) == []


def test_process_samples_crops_and_marks_repeat():
    calls = []

    def analyzer(crop):
        calls.append(crop.shape)
        return {"defect_class": "Crack", "confidence": 0.9, "scores": {}, "source": "fake"}

    frame = crack_frame()
    loc = FixedLocalizer([Box(100, 80, 200, 140, 0.5)])
    samples = [FrameSample(0, 0.0, frame), FrameSample(50, 2.0, frame)]
    events = process_samples(samples, loc, analyzer, pad_frac=0.2)
    assert len(events) == 2
    assert events[0].repeat is False and events[1].repeat is True
    assert len(calls) == 1                      # repeat not re-analyzed
    assert events[0].crop.shape[:2] == (events[0].padded_box[3] - events[0].padded_box[1],
                                        events[0].padded_box[2] - events[0].padded_box[0])
    assert events[0].annotated is not None


def test_min_score_filters():
    loc = FixedLocalizer([Box(10, 10, 60, 60, 0.1)])
    assert process_samples([FrameSample(0, 0.0, crack_frame())], loc, stub_analyzer, min_score=0.3) == []


@pytest.fixture
def avi_path(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    for _ in range(60):          # 6 s at 10 fps
        writer.write(crack_frame())
    writer.release()
    return path


def test_sample_frames_interval(avi_path):
    samples = list(sample_frames(avi_path, interval_s=2.0))
    assert [s.index for s in samples] == [0, 20, 40]
    assert [round(s.t_sec, 1) for s in samples] == [0.0, 2.0, 4.0]


def test_process_video_end_to_end(avi_path):
    events = process_video(avi_path, OpenCVAnomalyLocalizer(), stub_analyzer, interval_s=2.0)
    assert len(events) >= 1
    assert events[0].analysis["source"] == "stub_not_evaluated"


def test_stub_analyzer_default_and_loader(monkeypatch):
    monkeypatch.delenv("RODIC_ANALYZER", raising=False)
    assert is_stub(load_analyzer())
    monkeypatch.setenv("RODIC_ANALYZER", "pipeline_adapter:stub_analyzer")
    assert load_analyzer() is stub_analyzer


def test_pdf_report_is_written(tmp_path):
    frame = crack_frame()
    findings = []
    for i, (cls, width) in enumerate([("Crack", 2.1), ("Efflorescence", None), (None, None)], start=1):
        findings.append({
            "id": i, "frame_index": i * 20, "t_sec": i * 2.0, "box": (1, 2, 3, 4), "image": frame,
            "defect_class": cls, "confidence": 0.8 if cls else None, "crack_width_mm": width,
            "source": "test", "recommendation": recommend(cls, 0.8 if cls else None, width).to_dict(),
        })
    out = export_report(str(tmp_path / "r.pdf"), findings, {"site": "Test <bridge> & co"}, max_detail=2)
    data = open(out, "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 2000


def test_pdf_with_no_findings(tmp_path):
    out = export_report(str(tmp_path / "empty.pdf"), [], {})
    assert open(out, "rb").read().startswith(b"%PDF")
