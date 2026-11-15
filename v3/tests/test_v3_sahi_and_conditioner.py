import cv2
import numpy as np

from edge_drone_pipeline.frame_conditioner import (
    FrameConditioner,
    apply_clahe_lab,
    clipping_fractions,
    laplacian_variance,
)
from edge_drone_pipeline.sahi_rtdetr_engine import (
    Det,
    FunctionDetector,
    SahiEngine,
    generate_tiles,
    ios,
    iou,
    nmm,
    nms,
    tile_starts,
)


def test_tile_starts_last_tile_flush():
    assert tile_starts(500, 640, 0.2) == [0]
    starts = tile_starts(3840, 640, 0.2)
    assert starts[0] == 0 and starts[-1] == 3840 - 640
    assert all(b > a for a, b in zip(starts, starts[1:]))
    assert max(b - a for a, b in zip(starts, starts[1:])) <= 512   # stride at 20 % overlap


def test_tiles_cover_every_pixel_of_a_4k_frame():
    w, h = 3840, 2160
    cover = np.zeros((h, w), bool)
    tiles = generate_tiles(w, h, 640, 0.2)
    for x1, y1, x2, y2 in tiles:
        cover[y1:y2, x1:x2] = True
    assert cover.all()
    assert len(tiles) == 8 * 4


def test_iou_ios():
    a, b = Det(0, 0, 10, 10, 0.9), Det(2, 2, 6, 6, 0.8)
    assert ios(a, b) == 1.0                  # b sits inside a
    assert iou(a, b) < 0.2


def test_nmm_joins_a_crack_split_by_a_tile_border_but_nms_keeps_both_fragments():
    left, right = Det(100, 100, 300, 140, 0.8, "crack"), Det(200, 105, 500, 145, 0.7, "crack")
    merged = nmm([left, right], ios_thr=0.3)
    assert len(merged) == 1
    m = merged[0]
    assert (m.x1, m.x2) == (100, 500) and m.score == 0.8
    assert len(nms([left, right], iou_thr=0.5)) == 2          # NMS never joins fragments


def test_nmm_limit_long_thin_fragments_with_little_overlap_stay_separate():
    # Known limitation: the fragments of a long diagonal crack overlap only inside the
    # tile overlap strip, so IOS can fall below the merge threshold and they stay split.
    left, right = Det(100, 100, 300, 140, 0.8, "crack"), Det(280, 105, 500, 145, 0.7, "crack")
    assert len(nmm([left, right], ios_thr=0.5)) == 2


def test_nmm_reaches_a_fixed_point_when_a_big_box_covers_separate_fragments():
    a, b = Det(895, 695, 1152, 858, 0.9, "x"), Det(1536, 1075, 1906, 1306, 0.85, "x")
    big = Det(894, 696, 1908, 1308, 0.6, "x")
    out = nmm([a, b, big], ios_thr=0.5)
    assert len(out) == 1 and out[0].x1 == 894 and out[0].x2 == 1908


def test_nmm_never_merges_different_labels():
    out = nmm([Det(0, 0, 10, 10, 0.9, "crack"), Det(0, 0, 10, 10, 0.8, "spall")])
    assert len(out) == 2


def dark_box_detector(tile):
    mask = tile.max(axis=2) < 80
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return []
    return [Det(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 0.9, "crack")]


def test_engine_maps_tile_boxes_back_to_frame_coordinates_and_dedups_overlap():
    frame = np.full((2160, 3840, 3), 150, np.uint8)
    frame[1000:1030, 2000:2300] = 20           # object spans an overlap region of several tiles
    engine = SahiEngine(FunctionDetector(dark_box_detector), 640, 0.2, conf=0.25, merge_ios=0.5,
                        batch_size=4, full_frame_pass=False)
    dets = engine.run(frame)
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.x1 - 2000) <= 1 and abs(d.x2 - 2300) <= 1
    assert abs(d.y1 - 1000) <= 1 and abs(d.y2 - 1030) <= 1


def test_engine_full_frame_pass_runs_on_downscaled_frame():
    seen = []

    def fn(tile):
        seen.append(tile.shape[:2])
        return []

    SahiEngine(FunctionDetector(fn), 640, 0.2, full_frame_pass=True).run(np.zeros((2160, 3840, 3), np.uint8))
    assert (360, 640) in seen                   # 3840x2160 scaled to a 640 long side


def test_engine_filters_low_confidence():
    engine = SahiEngine(FunctionDetector(lambda t: [Det(0, 0, 10, 10, 0.1)]), 640, 0.2, conf=0.25,
                        full_frame_pass=False)
    assert engine.run(np.zeros((640, 640, 3), np.uint8)) == []


def test_laplacian_variance_separates_sharp_from_blurred():
    rng = np.random.default_rng(1)
    sharp = (rng.random((1080, 1920)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert laplacian_variance(sharp) > 100 > laplacian_variance(blurred)


def test_conditioner_rejects_blur_and_bad_exposure_and_accepts_good():
    rng = np.random.default_rng(2)
    good = cv2.cvtColor((rng.random((1080, 1920)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cond = FrameConditioner(blur_laplacian_min=100.0)
    out, rep = cond.process(good)
    assert rep.accepted and rep.clahe_applied

    blurry = cv2.GaussianBlur(good, (0, 0), 8)
    assert not cond.process(blurry)[1].accepted

    black = np.zeros((1080, 1920, 3), np.uint8)
    reasons = " ".join(cond.process(black)[1].reasons)
    assert "crushed" in reasons

    white = np.full((1080, 1920, 3), 255, np.uint8)
    assert "blown" in " ".join(cond.process(white)[1].reasons)


def test_clahe_lab_raises_contrast_of_a_dim_frame():
    rng = np.random.default_rng(3)
    dim = np.clip(60 + rng.normal(0, 4, (480, 640, 3)), 0, 255).astype(np.uint8)
    out = apply_clahe_lab(dim)
    assert out.shape == dim.shape and out.std() > dim.std()
    low, high = clipping_fractions(out)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
