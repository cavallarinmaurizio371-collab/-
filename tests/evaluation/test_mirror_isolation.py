from last_app import _display_point
from src.experimental_3d_pointing.coarse import map_coarse_region


def test_preview_mirror_cannot_change_business_region():
    region = map_coarse_region(-8, 0)
    assert _display_point((100, 50), 1280, True) != _display_point((100, 50), 1280, False)
    assert map_coarse_region(-8, 0) == region == "LEFT"
