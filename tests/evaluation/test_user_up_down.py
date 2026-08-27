from src.experimental_3d_pointing.coarse import camera_hit_to_eval, map_coarse_region


def test_camera_y_converts_to_user_physical_up_down():
    assert camera_hit_to_eval([0, -.2])[1] > 0
    assert map_coarse_region(0, 8) == "UP"
    assert map_coarse_region(0, -8) == "DOWN"
