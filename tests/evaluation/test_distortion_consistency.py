from types import SimpleNamespace

import numpy as np

from src.experimental_3d_pointing.intrinsics import (backproject_distorted_pixel,
    project_camera_points)


def test_projection_and_backprojection_share_distortion_model():
    intr=SimpleNamespace(fx=764.8,fy=770.6,cx=602.5,cy=296.6,
        distortion=[-.075,.161,.002,-.006,-.09])
    point=np.array([.08,-.04,.75])
    pixel=project_camera_points([point],intr)[0]
    restored=backproject_distorted_pixel(*pixel,point[2],intr)
    assert np.allclose(restored,point,atol=1e-7)

