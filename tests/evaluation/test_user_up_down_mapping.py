from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_raw_camera_vertical_maps_to_user_physical_up_down():
    up = normalized_camera_plane_hit([320, 240], [0, -.2, -1], INTRINSICS)
    down = normalized_camera_plane_hit([320, 240], [0, .2, -1], INTRINSICS)
    assert up.region == "UP"
    assert down.region == "DOWN"
