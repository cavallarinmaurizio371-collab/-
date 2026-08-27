import numpy as np

from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_no_object_is_forced_when_all_fail_thresholds():
    detected = ObjectDetection(1, 44, "bottle", .9, (600, 200, 650, 260), (625, 230),
                               center_camera=np.array([1.0, 0, 1.0]), depth_valid=True)
    selected, metric, _ = select_object_by_ray([detected], [0, 0, .5], [0, 0, 1], INTRINSICS,
                                                base_max_angle_deg=3)
    assert selected is None and metric is None
