import numpy as np

from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_side_pointing_ray_still_selects_lateral_object():
    detected = ObjectDetection(1, 47, "cup", .9, (500, 200, 580, 300), (540, 250),
                               center_camera=np.array([.5, 0, .8]), depth_valid=True)
    selected, _, _ = select_object_by_ray([detected], [0, 0, .8], [1, 0, 0],
                                           INTRINSICS, base_max_angle_deg=5)
    assert selected is detected
