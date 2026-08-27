from types import SimpleNamespace

import cv2
import numpy as np


INTRINSICS = SimpleNamespace(
    fx=800.0, fy=800.0, cx=320.0, cy=240.0,
    distortion=[0.0] * 5, mode="CALIBRATED_INTRINSICS",
)


def orientation_config():
    return {
        "current": {"max_reprojection_rmse_px": 25.0},
        "ransac": {
            "iterations_count": 100, "reprojection_error_px": 4.0,
            "confidence": .99, "minimum_inliers": 8,
            "minimum_inlier_rate": .45, "max_reprojection_rmse_px": 10.0,
        },
        "palm_stable": {
            "landmark_indices": [0, 1, 2, 5, 9, 13, 17],
            "iterations_count": 100, "reprojection_error_px": 4.0,
            "confidence": .99, "minimum_inliers": 5,
            "minimum_inlier_rate": .6, "max_reprojection_rmse_px": 10.0,
        },
        "selection": {
            "max_projection_agreement_deg": 35.0,
            "max_rotation_jump_deg": 30.0,
            "priority": ["R_PALM_STABLE", "R21_RANSAC", "R21_CURRENT"],
        },
    }


def synthetic_correspondences():
    # A deterministic non-planar hand-like cloud with stable palm indices.
    rng = np.random.default_rng(42)
    world = rng.normal(0.0, .025, (21, 3))
    world[[0, 1, 2, 5, 9, 13, 17], 2] += np.asarray([0, .005, .01, -.005, .008, -.009, .004])
    world[5] = [-.02, .02, .01]
    world[6] = [-.01, .005, -.015]
    world[7] = [0.0, -.01, -.04]
    world[8] = [.01, -.025, -.065]
    rvec = np.asarray([.08, -.12, .04], dtype=float)
    tvec = np.asarray([.01, -.015, .72], dtype=float)
    matrix = np.asarray([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    image, _ = cv2.projectPoints(world, rvec, tvec, matrix, np.zeros(5))
    rotation, _ = cv2.Rodrigues(rvec)
    return world, image.reshape(-1, 2), rotation
