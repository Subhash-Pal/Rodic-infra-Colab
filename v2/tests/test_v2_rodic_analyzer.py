import sys
import types

import numpy as np
import pytest

from rodic_analyzer import HostedPipeline, classify_scores

CLASSES = ["background", "crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]


def codebrim(**scores):
    return [{"defect": c, "score": scores.get(c, 0.05), "detected": scores.get(c, 0.05) >= 0.5} for c in CLASSES]


def test_top_codebrim_defect_over_threshold_wins():
    out = classify_scores(codebrim(crack=0.62, spallation=0.9), {"crack_probability": 0.1}, 0.5)
    assert out["defect_class"] == "spallation" and out["confidence"] == 0.9
    assert out["class_basis"] == "codebrim"
    assert [p["class"] for p in out["cv_predictions"]] == ["spallation", "crack"]


def test_background_is_never_reported_as_a_defect():
    out = classify_scores(codebrim(background=0.99), {"crack_probability": 0.1}, 0.5)
    assert out["defect_class"] == "background" and out["cv_predictions"] == []


def test_sdnet_alone_can_supply_a_crack_but_is_labelled_as_such():
    out = classify_scores(codebrim(), {"crack_probability": 0.8}, 0.5)
    assert out["defect_class"] == "crack" and out["class_basis"] == "sdnet_only"
    assert out["confidence"] == 0.8 and out["cv_predictions"] == []   # arbitration input stays CODEBRIM-only


def test_sources_are_never_averaged():
    out = classify_scores(codebrim(crack=0.9), {"crack_probability": 0.3}, 0.5)
    assert out["confidence"] == 0.9                                    # not (0.9 + 0.3) / 2
    assert out["scores"]["codebrim"]["crack"] == 0.9 and out["scores"]["sdnet_crack_probability"] == 0.3


def test_threshold_is_respected():
    scores = codebrim(crack=0.55)
    assert classify_scores(scores, {"crack_probability": 0.0}, 0.5)["defect_class"] == "crack"
    assert classify_scores(scores, {"crack_probability": 0.0}, 0.6)["defect_class"] == "background"


def test_hosted_pipeline_converts_bgr_to_rgb_and_uses_hosted_functions(monkeypatch):
    seen = {}

    def run_codebrim(model, rgb):
        seen["rgb_pixel"] = tuple(int(v) for v in rgb[0, 0])
        return codebrim(crack=0.7)

    fake = types.ModuleType("rodic_inference.inference")
    fake.load_codebrim_model = lambda: "cb-model"
    fake.load_sdnet_model = lambda: "sd-model"
    fake.run_codebrim_inference = run_codebrim
    fake.run_sdnet_inference = lambda model, rgb: {"crack_probability": 0.4, "crack_detected": False}
    pkg = types.ModuleType("rodic_inference")
    pkg.inference = fake
    monkeypatch.setitem(sys.modules, "rodic_inference", pkg)
    monkeypatch.setitem(sys.modules, "rodic_inference.inference", fake)

    ensured = []
    pipe = HostedPipeline(ensure=lambda: ensured.append(True))
    assert ensured == [True] and pipe.codebrim_model == "cb-model"

    bgr = np.zeros((40, 40, 3), np.uint8)
    bgr[..., 0], bgr[..., 2] = 10, 200                                  # B=10, R=200
    out = pipe.analyze(bgr, threshold=0.5)
    assert seen["rgb_pixel"] == (200, 0, 10)                            # hosted models expect RGB
    assert out["defect_class"] == "crack" and out["threshold"] == 0.5


def test_model_download_failure_propagates_from_hosted_pipeline():
    def broken():
        raise RuntimeError("download blocked")

    with pytest.raises(RuntimeError, match="download blocked"):
        HostedPipeline(ensure=broken)
