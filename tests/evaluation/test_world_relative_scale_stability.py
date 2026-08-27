import numpy as np

from src.experimental_3d_pointing.core import scale_stability,world_relative_scale
from tests.evaluation._phase2_helpers import synthetic_hand


def test_world_relative_scale_metrics_and_cv():
    world,_,_=synthetic_hand(); metrics=world_relative_scale(world,9)
    assert metrics["tip_anchor_distance_m"]>0 and metrics["finger_length_m"]>0
    stable=scale_stability([.100,.101,.099,.100])
    unstable=scale_stability([.08,.10,.13,.09])
    assert stable["cv"]<.01 and unstable["cv"]>.15
