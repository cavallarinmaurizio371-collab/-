from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_frontal_toward_camera_direction_produces_center():
    result = normalized_camera_plane_hit([320, 240], [0, 0, -1], INTRINSICS)
    assert result.valid and result.region == "CENTER"
