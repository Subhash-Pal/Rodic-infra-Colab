import math

import cv2
import numpy as np
import pytest

from edge_drone_pipeline.active_zoom_tracker import (
    LoggingGimbal,
    SimpleIoUTracker,
    pixel_offset_to_gimbal,
    roi_crop,
    unsharp,
)
from edge_drone_pipeline.sahi_rtdetr_engine import Det
from edge_drone_pipeline.sensorless_metrology import (
    gsd_from_distance,
    gsd_from_reference,
    gsd_needed_mm_per_px,
    measure_crack_width,
    measure_from_crop,
    min_measurable_width_mm,
    segment_crack,
)

BOX = Det(100, 100, 200, 160, 0.8, "crack")


def test_trigger_only_after_three_consecutive_frames_and_only_once():
    tr = SimpleIoUTracker(iou_match=0.3, confirm_frames=3, max_age=5)
    assert tr.update([BOX])[1] == []
    assert tr.update([BOX])[1] == []
    _, fired = tr.update([BOX])
    assert len(fired) == 1
    assert tr.update([BOX])[1] == []            # no second trigger for the same track


def test_gap_resets_the_consecutive_counter():
    tr = SimpleIoUTracker(confirm_frames=3, max_age=5)
    tr.update([BOX]); tr.update([BOX])
    tr.update([])                                # missed frame
    assert tr.update([BOX])[1] == []
    assert tr.update([BOX])[1] == []
    assert len(tr.update([BOX])[1]) == 1


def test_old_tracks_are_dropped_and_moving_box_keeps_its_track():
    tr = SimpleIoUTracker(confirm_frames=3, max_age=2)
    tr.update([BOX])
    for _ in range(3):
        tr.update([])
    assert tr.tracks == {}
    for dx in (0, 8, 16):                        # slow drift keeps IoU above the threshold
        tracks, fired = tr.update([Det(100 + dx, 100, 200 + dx, 160, 0.8, "crack")])
    assert len(tracks) == 1 and len(fired) == 1


def test_gimbal_pinhole_math():
    c = pixel_offset_to_gimbal((450, 250, 550, 350), 1000, 600, hfov_deg=90)   # centred
    assert c.yaw_deg == 0 and c.pitch_deg == 0
    right = pixel_offset_to_gimbal((950, 250, 1050, 350), 1000, 600, hfov_deg=90)
    assert right.yaw_deg == pytest.approx(45.0, abs=0.01)                       # f = 500 px, dx = 500
    up = pixel_offset_to_gimbal((450, 0, 550, 100), 1000, 600, hfov_deg=90)
    assert up.pitch_deg == pytest.approx(math.degrees(math.atan(250 / 500)), abs=0.01) and up.pitch_deg > 0
    left = pixel_offset_to_gimbal((0, 250, 100, 350), 1000, 600, hfov_deg=90)
    assert left.yaw_deg < 0


def test_gimbal_deadband_and_zoom_clamp():
    near = pixel_offset_to_gimbal((502, 303, 512, 313), 1000, 600, 90, deadband_deg=1.0)   # dx = 7 px = 0.8 deg
    assert near.yaw_deg == 0 and near.pitch_deg == 0
    assert near.zoom_factor == 5.0               # tiny box -> clamped to max optical zoom
    big = pixel_offset_to_gimbal((0, 0, 1000, 600), 1000, 600, 90)
    assert big.zoom_factor == 1.0                # already fills the frame -> no zoom


def test_logging_gimbal_only_records():
    g = LoggingGimbal()
    g.send(pixel_offset_to_gimbal((0, 0, 50, 50), 1000, 600, 90))
    assert len(g.sent) == 1


def test_roi_crop_padding_bounds_and_sharpen():
    rng = np.random.default_rng(0)
    frame = (rng.random((2160, 3840, 3)) * 255).astype(np.uint8)
    crop, padded = roi_crop(frame, (1000, 1000, 1100, 1060), padding=0.25, unsharp_amount=0.0)
    assert padded == (975, 985, 1125, 1075)
    assert crop.shape[:2] == (90, 150) and np.array_equal(crop, frame[985:1075, 975:1125])
    edge, padded_edge = roi_crop(frame, (0, 0, 40, 40), padding=0.25, unsharp_amount=0.6)
    assert padded_edge[0] == 0 and padded_edge[1] == 0
    assert unsharp(crop, 0.6).shape == crop.shape


def test_gsd_formulas():
    assert gsd_from_distance(10, 3.4, 8.8) == pytest.approx(3.8636, abs=1e-3)
    assert gsd_from_distance(5, 3.4, 8.8) == pytest.approx(gsd_from_distance(10, 3.4, 8.8) / 2)
    assert gsd_from_reference(100.0, 50.0) == 2.0
    with pytest.raises(ValueError):
        gsd_from_distance(0, 3.4, 8.8)


def test_resolution_limit_statement_matches_the_numbers():
    gsd = gsd_from_distance(10, 3.4, 8.8)
    assert min_measurable_width_mm(gsd) > 7        # at 10 m a 0.2 mm or 2 mm crack is NOT measurable
    assert gsd_needed_mm_per_px(0.2) == pytest.approx(0.1)


def crack_mask(thickness, h=200, w=400):
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, (20, 40), (380, 160), 255, thickness)
    return m


def test_measures_a_known_width():
    m = measure_crack_width(crack_mask(7), gsd_mm_per_px=0.3, distance_rel_error=0.2)
    assert m.resolvable
    assert 5.5 <= m.width_px_p95 <= 8.0
    assert m.width_mm_p95 == pytest.approx(m.width_px_p95 * 0.3)
    low, high = m.interval_mm
    assert low < m.width_mm_for_decision < high
    assert high == pytest.approx(m.width_mm_for_decision * 1.2)


def test_width_against_rendered_ground_truth_area_over_length():
    # OpenCV draws thick diagonal lines wider than the nominal thickness, so the truth is the
    # rendered mask's area / length, not the 'thickness' argument.
    for thickness in (3, 5, 9):
        mask = crack_mask(thickness)
        truth = (mask > 0).sum() / np.hypot(360, 120)
        m = measure_crack_width(mask, gsd_mm_per_px=1.0)
        assert abs(m.width_px_median - truth) / truth <= 0.2, (thickness, m.width_px_median, truth)


def test_sub_pixel_crack_is_an_upper_bound_not_a_measurement():
    m = measure_crack_width(crack_mask(1), gsd_mm_per_px=3.9)
    assert not m.resolvable
    assert m.width_mm_for_decision == pytest.approx(2 * 3.9)
    assert "upper bound" in m.note


def test_empty_mask_returns_none():
    assert measure_crack_width(np.zeros((50, 50), np.uint8), 1.0) is None


def test_segment_and_measure_from_image_crop():
    rng = np.random.default_rng(4)
    img = np.clip(150 + rng.normal(0, 2, (200, 400)), 0, 255).astype(np.uint8)
    cv2.line(img, (20, 40), (380, 160), 30, 7)
    mask = segment_crack(img)
    assert mask.sum() > 0
    m = measure_from_crop(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.3)
    assert m is not None and 5.0 <= m.width_px_p95 <= 9.0
