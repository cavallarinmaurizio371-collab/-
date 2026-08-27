from src.experimental_3d_pointing.coarse import camera_hit_to_eval, map_coarse_region


def test_camera_x_converts_to_user_physical_left_right():
    assert camera_hit_to_eval([-.2, 0])[0] > 0
    assert map_coarse_region(8, 0) == "RIGHT"
    assert map_coarse_region(-8, 0) == "LEFT"
