import numpy as np

from src.experimental_3d_pointing.core import camera_ray


def test_camera_ray_has_tip_origin_and_unit_direction():
    origin,direction=camera_ray([.1,.2,.7],[0,0,-3])
    assert np.allclose(origin,[.1,.2,.7])
    assert np.allclose(direction,[0,0,-1])
