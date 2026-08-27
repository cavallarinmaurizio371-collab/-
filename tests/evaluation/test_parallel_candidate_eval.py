from types import SimpleNamespace

import numpy as np

from evaluation.phase2b_metrics import evaluate_camera_candidate


def test_parallel_candidates_are_evaluated_on_same_frame():
    origin=np.array([.01,.02,.7])
    b=SimpleNamespace(origin_camera=origin,smoothed_direction_camera=np.array([0.,0.,-1.]),quality="GOOD")
    c=SimpleNamespace(origin_camera=origin,smoothed_direction_camera=np.array([-.01,-.02,-1.]),quality="GOOD")
    targets={"CENTER":(0.,0.)}
    bh=evaluate_camera_candidate("B",b,targets,(0.,0.),210,297)
    ch=evaluate_camera_candidate("C",c,targets,(0.,0.),210,297)
    assert bh.intersection_valid and ch.intersection_valid
    assert not np.allclose(bh.hit_eval_mm,ch.hit_eval_mm)

