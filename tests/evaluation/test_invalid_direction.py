from evaluation.z0_geometry import intersect_ray_with_z0


def test_parallel_ray_is_rejected():
    result=intersect_ray_with_z0([0,0,1],[1,0,1e-12])
    assert not result.valid
    assert result.status=="NEAR_PARALLEL"


def test_ray_pointing_away_is_not_reversed():
    result=intersect_ray_with_z0([0,0,1],[0,0,1])
    assert not result.valid
    assert result.status=="POINTING_AWAY_FROM_CAMERA"
    assert result.t_hit<0
