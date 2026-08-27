from src.experimental_3d_pointing.coarse import map_coarse_region


def test_all_four_corners():
    assert map_coarse_region(-8, 8) == "LEFT_UP"
    assert map_coarse_region(8, 8) == "RIGHT_UP"
    assert map_coarse_region(-8, -8) == "LEFT_DOWN"
    assert map_coarse_region(8, -8) == "RIGHT_DOWN"
