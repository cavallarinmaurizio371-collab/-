from evaluation.coordinate_adapter import mirror_display_x
from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_mirror_changes_only_rendered_x_not_region():
    hit_before = normalized_camera_plane_hit([320, 240], [-.2, 0, -1], INTRINSICS)
    assert mirror_display_x(100, 1280, True) != mirror_display_x(100, 1280, False)
    hit_after = normalized_camera_plane_hit([320, 240], [-.2, 0, -1], INTRINSICS)
    assert hit_before.region == hit_after.region == "RIGHT"
