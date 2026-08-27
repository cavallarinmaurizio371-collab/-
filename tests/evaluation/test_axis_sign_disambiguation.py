import numpy as np

from evaluation.direction_candidates import fit_finger_axis_with_quality
from tests.evaluation._phase2_helpers import synthetic_hand


def test_axis_sign_always_follows_mcp_to_tip():
    world,_,_=synthetic_hand(); result=fit_finger_axis_with_quality(world)
    assert np.dot(result.direction,world[8]-world[5])>0
