import numpy as np

from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_same_camera_ray_drives_object_and_coarse_outputs():
    origin = np.array([0.0, 0.0, .7]); direction = np.array([0.0, 0.0, -1.0])
    detected = ObjectDetection(1, 44, "bottle", .9, (290, 210, 350, 270), (320, 240),
                               center_camera=np.array([0.0, 0.0, .3]), depth_valid=True)
    selected, _, _ = select_object_by_ray([detected], origin, direction, INTRINSICS)
    hit = normalized_camera_plane_hit([320, 240], direction, INTRINSICS)
    assert selected is detected and hit.region == "CENTER"
