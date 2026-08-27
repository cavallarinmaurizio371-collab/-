from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_finite_toward_direction_always_has_infinite_plane_hit():
    hit = normalized_camera_plane_hit([400, 260], [-.2, .1, -1], INTRINSICS)
    assert hit.valid and hit.region is not None and hit.status == "VALID"
