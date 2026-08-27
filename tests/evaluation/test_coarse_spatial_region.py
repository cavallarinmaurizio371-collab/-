from src.experimental_3d_pointing.coarse import map_coarse_region


def test_spatial_region_includes_center_cardinals_and_corners():
    expected = {"CENTER", "LEFT", "RIGHT", "UP", "DOWN", "LEFT_UP",
                "RIGHT_UP", "LEFT_DOWN", "RIGHT_DOWN"}
    actual = {map_coarse_region(x, y) for x in (-8, 0, 8) for y in (-8, 0, 8)}
    assert actual == expected
