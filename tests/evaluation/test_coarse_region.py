from src.experimental_3d_pointing.coarse import map_coarse_region


def test_infinite_plane_maps_all_finite_angles_to_a_region():
    values = [map_coarse_region(yaw, pitch) for yaw in (-80, 0, 80)
              for pitch in (-80, 0, 80)]
    assert all(values) and len(set(values)) == 9
