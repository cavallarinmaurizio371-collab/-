import numpy as np

from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_excessive_object_angle_is_rejected():
    detected = ObjectDetection(1, 44, "bottle", .9, (300, 220, 340, 260), (320, 240),
        center_camera=np.array([.8, 0, 1.0]), depth_valid=True)
    selected, _, _ = select_object_by_ray([detected], [0, 0, .5], [0, 0, 1],
        INTRINSICS, base_max_angle_deg=2, max_perpendicular_distance_m=2)
    assert selected is None
