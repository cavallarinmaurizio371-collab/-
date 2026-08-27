import numpy as np

from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def _object(identifier, xyz):
    return ObjectDetection(identifier, 44, "bottle", .9, (300, 220, 340, 260), (320, 240),
                           center_camera=np.asarray(xyz, dtype=float), depth_valid=True)


def test_ray_aligned_far_object_beats_closer_angularly_offset_object():
    close_offset = _object(1, [.12, 0, .9]); far_aligned = _object(2, [0, 0, 1.4])
    selected, _, _ = select_object_by_ray([close_offset, far_aligned], [0, 0, .5], [0, 0, 1],
        INTRINSICS, base_max_angle_deg=15, max_perpendicular_distance_m=.3)
    assert selected.id == 2


def test_ray_facing_object_a_selects_a_over_offset_b():
    aligned = _object(1, [0, 0, 1.0]); offset = _object(2, [.15, 0, 1.0])
    selected, _, _ = select_object_by_ray([aligned, offset], [0, 0, .5], [0, 0, 1], INTRINSICS)
    assert selected.id == 1
