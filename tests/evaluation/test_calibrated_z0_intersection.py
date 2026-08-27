from types import SimpleNamespace

import numpy as np

from evaluation.phase2b_metrics import evaluate_camera_candidate


def test_calibrated_camera_ray_hits_z0_center():
    candidate=SimpleNamespace(origin_camera=np.array([0.,0.,.7]),
        smoothed_direction_camera=np.array([0.,0.,-1.]),quality="GOOD")
    result=evaluate_camera_candidate("C",candidate,{"CENTER":(0.,0.)},(0.,0.),210,297)
    assert result.intersection_valid and result.in_target_range
    assert result.pred_region=="CENTER"
    assert np.allclose(result.hit_eval_mm,[0.,0.])

