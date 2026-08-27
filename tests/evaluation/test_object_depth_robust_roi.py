import numpy as np

from src.experimental_3d_pointing.generic_detector import robust_bbox_depth


def test_central_roi_median_rejects_background_and_outlier_depth():
    depth = np.full((100, 100), 5.0, dtype=float)
    depth[30:70, 30:70] = 1.2
    depth[45, 45] = 99.0; depth[46, 46] = 0.0
    result = robust_bbox_depth(depth, (20, 20, 80, 80), inner_ratio=.5)
    assert result.valid and abs(result.median_m - 1.2) < 1e-6
