from src.experimental_3d_pointing.coarse import map_coarse_region


def test_cardinal_coarse_regions():
    assert map_coarse_region(0, 0) == "CENTER"
    assert map_coarse_region(-8, 0) == "LEFT"
    assert map_coarse_region(8, 0) == "RIGHT"
    assert map_coarse_region(0, 8) == "UP"
    assert map_coarse_region(0, -8) == "DOWN"
