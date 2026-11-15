"""sensorless_metrology.py - crack width in millimetres from one RGB image.

Two steps:
1. Scale: ground sample distance (GSD), the real-world size of one pixel.
       GSD [mm/px] = D [m] x p [um] / f [mm]
   with D = camera-to-surface distance, p = sensor pixel pitch, f = focal length.
   (Units: D*1000 mm x p/1000 mm / f mm.) A known reference object in the frame
   gives the scale directly and does not need D at all - prefer it when you can.
2. Width: segment the crack, skeletonize it, and read the local width from the
   distance transform: width_px = 2 x dist_at_skeleton - 1.

THE LIMIT YOU MUST STATE TO ANY JURY
------------------------------------
Width cannot be measured below roughly 2 pixels. With D = 10 m, p = 3.4 um and
f = 8.8 mm the GSD is about 3.9 mm/px, so a 0.2 mm hairline crack is far below
one pixel and a 2 mm crack is about half a pixel. At that range only wide cracks
can be sized; to size a 0.2 mm crack you need a much smaller GSD (closer, longer
focal length / optical zoom) - e.g. ~0.1 mm/px. When the measured width is below
``min_resolvable_px`` this module reports an UPPER BOUND, not a width, and flags
it ``resolvable=False``.

Distance from monocular depth (e.g. Depth-Anything-V2) is a plug-in via the
``DistanceEstimator`` protocol: relative-depth outputs are not metric, so only a
metric-trained model or a reference-scaled one is usable. None is bundled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def gsd_from_distance(distance_m: float, pixel_pitch_um: float, focal_mm: float) -> float:
    if distance_m <= 0 or pixel_pitch_um <= 0 or focal_mm <= 0:
        raise ValueError("distance, pixel pitch and focal length must be positive.")
    return distance_m * pixel_pitch_um / focal_mm


def gsd_from_reference(ref_size_mm: float, ref_size_px: float) -> float:
    if ref_size_mm <= 0 or ref_size_px <= 0:
        raise ValueError("reference sizes must be positive.")
    return ref_size_mm / ref_size_px


def min_measurable_width_mm(gsd_mm_per_px: float, min_resolvable_px: float = 2.0) -> float:
    return gsd_mm_per_px * min_resolvable_px


def gsd_needed_mm_per_px(target_width_mm: float, min_resolvable_px: float = 2.0) -> float:
    """GSD required to resolve a crack of ``target_width_mm``."""
    return target_width_mm / min_resolvable_px


class DistanceEstimator(Protocol):
    def distance_m(self, frame_bgr: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[float]: ...


class ConstantDistance:
    def __init__(self, distance_m: float) -> None:
        self.value = distance_m

    def distance_m(self, frame_bgr, box):
        return self.value


def segment_crack(gray: np.ndarray, block: int = 31, c: float = 6.0, min_area: int = 20) -> np.ndarray:
    """Dark, thin structures on a lighter surface -> boolean mask of the largest
    connected component. Heuristic; replace with a trained segmentation model for
    real use (deliverable D10)."""
    block = max(3, block | 1)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Raise the threshold offset on noisy / textured surfaces: robust noise estimate (MAD)
    # of the residual against a heavily smoothed copy.
    residual = blurred.astype(np.float32) - cv2.GaussianBlur(blurred, (0, 0), 8).astype(np.float32)
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    c = max(c, 3.0 * sigma)
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros(gray.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if areas.max() < min_area:
        return np.zeros(gray.shape, bool)
    return labels == best


@dataclass
class CrackMeasurement:
    gsd_mm_per_px: float
    length_px: int
    width_px_median: float
    width_px_p95: float
    width_mm_median: float
    width_mm_p95: float
    resolvable: bool
    width_mm_for_decision: float      # p95 if resolvable, else an upper bound
    interval_mm: Optional[Tuple[float, float]] = None
    note: str = ""


def measure_crack_width(
    mask: np.ndarray,
    gsd_mm_per_px: float,
    min_resolvable_px: float = 2.0,
    distance_rel_error: Optional[float] = None,
) -> Optional[CrackMeasurement]:
    """Width statistics along the crack skeleton. Returns None for an empty mask.

    The decision value uses the 95th percentile (conservative: the widest part
    matters for severity), not the median.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return None
    skeleton = skeletonize(mask_u8 > 0)
    if skeleton.sum() == 0:
        return None
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    widths_px = np.clip(2.0 * dist[skeleton] - 1.0, 1.0, None)
    median_px = float(np.median(widths_px))
    p95_px = float(np.percentile(widths_px, 95))
    resolvable = p95_px >= min_resolvable_px
    p95_mm = p95_px * gsd_mm_per_px
    if resolvable:
        decision, note = p95_mm, ""
    else:
        decision = min_resolvable_px * gsd_mm_per_px
        note = (f"Crack is thinner than {min_resolvable_px:g} px at this GSD "
                f"({gsd_mm_per_px:.2f} mm/px): width is an upper bound (< {decision:.2f} mm), not a measurement.")
    interval = None
    if distance_rel_error is not None:
        interval = (decision * (1.0 - distance_rel_error), decision * (1.0 + distance_rel_error))
    return CrackMeasurement(
        gsd_mm_per_px, int(skeleton.sum()), median_px, p95_px,
        median_px * gsd_mm_per_px, p95_mm, resolvable, float(decision), interval, note,
    )


def measure_from_crop(
    crop_bgr: np.ndarray,
    gsd_mm_per_px: float,
    min_resolvable_px: float = 2.0,
    distance_rel_error: Optional[float] = None,
) -> Optional[CrackMeasurement]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return measure_crack_width(segment_crack(gray), gsd_mm_per_px, min_resolvable_px, distance_rel_error)
