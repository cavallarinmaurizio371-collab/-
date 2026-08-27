import numpy as np

import last_app
from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_last_app_mirror_helpers_do_not_enter_geometry():
    direction = np.array([-.2, 0, -1.0])
    before = normalized_camera_plane_hit([320, 240], direction, INTRINSICS)
    assert last_app._display_point((100, 50), 1280, True)[0] == 1179
    after = normalized_camera_plane_hit([320, 240], direction, INTRINSICS)
    assert before.region == after.region == "RIGHT"
