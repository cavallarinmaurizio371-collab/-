import numpy as np

from src.experimental_3d_pointing.core import ExperimentalPointingCore
from tests.evaluation._phase2_helpers import INTRINSICS,phase2_config,synthetic_hand


def test_candidate_b_and_c_use_identical_tip_origin():
    world,pixels,_=synthetic_hand(); depth=np.full((480,640),.7,np.float32)
    result=ExperimentalPointingCore(phase2_config()).process(world,pixels,depth,INTRINSICS)
    b=result.candidates["B_RELATIVE"]; c=result.candidates["C_AXIS_FIT"]
    assert b.origin_camera is not None and c.origin_camera is not None
    assert np.array_equal(b.origin_camera,c.origin_camera)

