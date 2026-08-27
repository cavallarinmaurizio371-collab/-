import numpy as np

from evaluation.direction_candidates import fit_finger_axis_with_quality
from tests.evaluation._phase2_helpers import synthetic_hand


def test_axis_fit_returns_direction_and_quality():
    world,_,_=synthetic_hand(); result=fit_finger_axis_with_quality(world)
    assert result.valid and np.isclose(np.linalg.norm(result.direction),1)
    assert result.residual_m is not None and result.residual_m<.012
    assert 0<=result.linearity<=1
