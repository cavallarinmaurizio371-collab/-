import numpy as np

from src.experimental_3d_pointing.coarse import normalized_camera_plane_hit
from tests.evaluation._mentor_helpers import INTRINSICS


def test_normalized_hit_ignores_scene_depth_and_anchor_kwargs():
    first = normalized_camera_plane_hit([320, 240], [-.1, 0, -1], INTRINSICS,
                                        scene_depth=0.4, anchor_xyz=[1, 2, 3])
    second = normalized_camera_plane_hit([320, 240], [-.1, 0, -1], INTRINSICS,
                                         scene_depth=9.0, anchor_xyz=[9, 8, 7])
    assert np.allclose(first.eval_xy, second.eval_xy)
