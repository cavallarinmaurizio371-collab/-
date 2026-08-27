from src.experimental_3d_pointing.coarse import map_coarse_region


def test_four_degree_boundary_is_inclusive_center():
    assert map_coarse_region(4.0, -4.0, 4.0, 4.0) == "CENTER"
    assert map_coarse_region(4.01, 0, 4.0, 4.0) == "RIGHT"
