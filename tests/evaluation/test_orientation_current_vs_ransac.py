from src.experimental_3d_pointing.orientation import solve_current, solve_ransac21
from tests.evaluation._mentor_helpers import INTRINSICS, orientation_config, synthetic_correspondences


def test_ransac_rejects_one_large_image_outlier():
    world, image, _ = synthetic_correspondences()
    image[20] += [140, -100]
    config = orientation_config()
    current = solve_current(world, image, INTRINSICS, config["current"])
    ransac = solve_ransac21(world, image, INTRINSICS, config["ransac"])
    assert ransac.valid and ransac.inlier_count < 21
    assert not current.valid or ransac.reprojection_rmse_px < current.reprojection_rmse_px
