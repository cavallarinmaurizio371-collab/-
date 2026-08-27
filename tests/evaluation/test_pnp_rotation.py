import numpy as np

from evaluation.direction_candidates import estimate_hand_to_camera_rotation
from tests.evaluation._phase2_helpers import INTRINSICS,synthetic_hand


def test_known_rotation_is_recovered_from_all_landmarks():
    world,pixels,expected=synthetic_hand()
    result=estimate_hand_to_camera_rotation(world,pixels,INTRINSICS,25)
    assert result.valid and result.reprojection_rmse_px<1e-3
    assert np.allclose(result.rotation,expected,atol=1e-3)
