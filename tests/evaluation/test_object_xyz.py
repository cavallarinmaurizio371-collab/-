import numpy as np

from src.experimental_3d_pointing.intrinsics import backproject_distorted_pixel
from tests.evaluation._mentor_helpers import INTRINSICS


def test_object_center_depth_backprojects_to_camera_xyz():
    point = backproject_distorted_pixel(320, 240, .8, INTRINSICS)
    assert np.allclose(point, [0, 0, .8])
