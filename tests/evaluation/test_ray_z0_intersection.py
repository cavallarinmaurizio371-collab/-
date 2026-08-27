import numpy as np

from evaluation.z0_geometry import intersect_ray_with_z0


def test_known_ray_hits_expected_point():
    result=intersect_ray_with_z0([0.1,0.2,1.0],[-0.1,-0.2,-1.0])
    assert result.valid
    assert result.status=="VALID"
    assert np.allclose(result.point_camera,[0,0,0])
    assert result.t_hit>0


def test_noncentral_hit():
    result=intersect_ray_with_z0([0.2,-0.1,1.0],[-0.1,0.0,-1.0])
    assert result.valid
    assert np.allclose(result.point_camera,[0.1,-0.1,0],atol=1e-6)

