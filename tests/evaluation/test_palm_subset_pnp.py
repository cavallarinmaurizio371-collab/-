import numpy as np

from src.experimental_3d_pointing.orientation import solve_palm_stable
from tests.evaluation._mentor_helpers import INTRINSICS, orientation_config, synthetic_correspondences


def test_palm_subset_excludes_index_distal_landmarks_and_recovers_rotation():
    world, image, expected = synthetic_correspondences()
    image[[6, 7, 8]] += [100, 80]
    result = solve_palm_stable(world, image, INTRINSICS, orientation_config()["palm_stable"])
    assert result.valid
    assert not set((6, 7, 8)) & set(result.landmark_indices)
    assert np.allclose(result.rotation, expected, atol=1e-3)
