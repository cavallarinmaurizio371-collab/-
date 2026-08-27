from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_raw_camera_left_is_user_physical_right_for_front_facing_camera():
    right = normalized_camera_plane_hit([320, 240], [-.2, 0, -1], INTRINSICS)
    left = normalized_camera_plane_hit([320, 240], [.2, 0, -1], INTRINSICS)
    assert right.region == "RIGHT"
    assert left.region == "LEFT"
