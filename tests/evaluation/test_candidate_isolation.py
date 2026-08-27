import numpy as np

import src.experimental_3d_pointing.core as core_module
from src.experimental_3d_pointing.core import ExperimentalPointingCore
from tests.evaluation._phase2_helpers import INTRINSICS,phase2_config,synthetic_hand


def test_candidate_c_direction_does_not_depend_on_tip_pip_scene_depth(monkeypatch):
    world,pixels,_=synthetic_hand(); depth=np.full((480,640),.7,np.float32)
    def run(tip_depth,pip_depth):
        def fake_sample(_map,point,_patch):
            if np.allclose(point,pixels[8,:2]): return tip_depth
            if np.allclose(point,pixels[6,:2]): return pip_depth
            return .70
        monkeypatch.setattr(core_module,"sample_point",fake_sample)
        return ExperimentalPointingCore(phase2_config()).process(world,pixels,depth,INTRINSICS)
    first=run(.5,.9); second=run(.9,.5)
    assert np.allclose(first.candidates["C_AXIS_FIT"].raw_direction_camera,
                       second.candidates["C_AXIS_FIT"].raw_direction_camera)
    assert np.sign(first.candidates["A_BASELINE"].raw_direction_camera[2]) != \
           np.sign(second.candidates["A_BASELINE"].raw_direction_camera[2])
