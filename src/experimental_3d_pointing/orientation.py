from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from src.experimental_3d_pointing.intrinsics import (
    camera_matrix,
    distortion_coefficients,
)


R21_CURRENT = "R21_CURRENT"
R21_RANSAC = "R21_RANSAC"
R_PALM_STABLE = "R_PALM_STABLE"
RAW_WORLD_DIAGNOSTIC = "RAW_WORLD_DIAGNOSTIC"


@dataclass(frozen=True)
class OrientationEstimate:
    name: str
    valid: bool
    status: str
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    rvec: np.ndarray | None = None
    reprojection_rmse_px: float | None = None
    inlier_count: int | None = None
    inlier_rate: float | None = None
    rotation_magnitude_deg: float | None = None
    landmark_indices: tuple[int, ...] = ()
    projection_agreement: dict[str, dict[str, float | None]] = field(default_factory=dict)


def _finite_points(world_landmarks, image_landmarks_px):
    world = np.asarray(world_landmarks, dtype=np.float64)
    image = np.asarray(image_landmarks_px, dtype=np.float64)
    if world.shape != (21, 3) or image.shape not in ((21, 2), (21, 3)):
        raise ValueError("Expected 21 world and image landmarks")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(image[:, :2])):
        raise ValueError("Landmarks must be finite")
    return np.ascontiguousarray(world), np.ascontiguousarray(image[:, :2])


def _rotation_magnitude(rotation):
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def rotation_geodesic_deg(first, second):
    if first is None or second is None:
        return None
    delta = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    return _rotation_magnitude(delta)


def _reprojection_rmse(world, image, indices, rvec, tvec, intrinsics):
    selected = np.asarray(indices, dtype=int)
    projected, _ = cv2.projectPoints(
        np.ascontiguousarray(world[selected]),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        camera_matrix(intrinsics),
        distortion_coefficients(intrinsics),
    )
    residual = projected.reshape(-1, 2) - image[selected]
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _finish(name, world, image, indices, success, rvec, tvec, intrinsics,
            max_rmse, inliers=None, min_inliers=0, min_inlier_rate=0.0):
    if not success:
        return OrientationEstimate(name, False, "PNP_FAILED", landmark_indices=tuple(indices))
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=float).reshape(3)
    count = None if inliers is None else int(len(inliers))
    rate = None if count is None else float(count / len(indices))
    if count:
        local = np.asarray(inliers, dtype=int).reshape(-1)
        rmse_indices = tuple(np.asarray(indices, dtype=int)[local].tolist())
    else:
        rmse_indices = indices
    rmse = _reprojection_rmse(world, image, rmse_indices, rvec, tvec, intrinsics)
    base = dict(rotation=rotation, translation=translation,
                rvec=np.asarray(rvec, dtype=float).reshape(3),
                reprojection_rmse_px=rmse, inlier_count=count, inlier_rate=rate,
                rotation_magnitude_deg=_rotation_magnitude(rotation),
                landmark_indices=tuple(int(v) for v in indices))
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        return OrientationEstimate(name, False, "PNP_NON_FINITE", **base)
    if translation[2] <= 0:
        return OrientationEstimate(name, False, "PNP_BEHIND_CAMERA", **base)
    if rmse > float(max_rmse):
        return OrientationEstimate(name, False, "PNP_HIGH_REPROJECTION_ERROR", **base)
    if count is not None and (count < int(min_inliers) or rate < float(min_inlier_rate)):
        return OrientationEstimate(name, False, "PNP_TOO_FEW_INLIERS", **base)
    return OrientationEstimate(name, True, "VALID", **base)


def solve_current(world_landmarks, image_landmarks_px, intrinsics, config):
    try:
        world, image = _finite_points(world_landmarks, image_landmarks_px)
        indices = tuple(range(21))
        success, rvec, tvec = cv2.solvePnP(
            world, image, camera_matrix(intrinsics), distortion_coefficients(intrinsics),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        return _finish(R21_CURRENT, world, image, indices, success, rvec, tvec,
                       intrinsics, config["max_reprojection_rmse_px"])
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return OrientationEstimate(R21_CURRENT, False, "PNP_INVALID_INPUT")


def _solve_ransac(name, world_landmarks, image_landmarks_px, intrinsics, config, indices):
    try:
        world, image = _finite_points(world_landmarks, image_landmarks_px)
        selected = np.asarray(indices, dtype=int)
        success, rvec, tvec, local_inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(world[selected]),
            np.ascontiguousarray(image[selected]),
            camera_matrix(intrinsics), distortion_coefficients(intrinsics),
            iterationsCount=int(config["iterations_count"]),
            reprojectionError=float(config["reprojection_error_px"]),
            confidence=float(config["confidence"]),
            flags=cv2.SOLVEPNP_EPNP,
        )
        inliers = np.empty((0, 1), dtype=int) if local_inliers is None else local_inliers
        return _finish(name, world, image, tuple(indices), success, rvec, tvec,
                       intrinsics, config["max_reprojection_rmse_px"], inliers,
                       config["minimum_inliers"], config["minimum_inlier_rate"])
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return OrientationEstimate(name, False, "PNP_INVALID_INPUT",
                                   landmark_indices=tuple(indices))


def solve_ransac21(world_landmarks, image_landmarks_px, intrinsics, config):
    return _solve_ransac(R21_RANSAC, world_landmarks, image_landmarks_px,
                         intrinsics, config, tuple(range(21)))


def solve_palm_stable(world_landmarks, image_landmarks_px, intrinsics, config):
    indices = tuple(int(v) for v in config["landmark_indices"])
    ransac = _solve_ransac(R_PALM_STABLE, world_landmarks, image_landmarks_px,
                           intrinsics, config, indices)
    if ransac.valid:
        return ransac
    # A seven-point palm subset is sometimes too small for RANSAC consensus in
    # frontal views. A single deterministic SQPnP fallback is allowed here;
    # it uses the same fixed subset and is selected only by reprojection
    # validity, never by target labels.
    try:
        world, image = _finite_points(world_landmarks, image_landmarks_px)
        selected = np.asarray(indices, dtype=int)
        success, rvec, tvec = cv2.solvePnP(
            np.ascontiguousarray(world[selected]), np.ascontiguousarray(image[selected]),
            camera_matrix(intrinsics), distortion_coefficients(intrinsics),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        direct = _finish(R_PALM_STABLE, world, image, indices, success, rvec, tvec,
                         intrinsics, config["max_reprojection_rmse_px"])
        return replace(direct, status="VALID_DIRECT_SUBSET") if direct.valid else ransac
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return ransac


def raw_world_orientation():
    return OrientationEstimate(
        RAW_WORLD_DIAGNOSTIC, True, "DIAGNOSTIC_ONLY",
        rotation=np.eye(3), rotation_magnitude_deg=0.0,
    )


def _normalize(vector):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value)) if value.shape == (3,) else 0.0
    return value / norm if norm > 1e-9 and np.all(np.isfinite(value)) else None


def direction_to_camera(native_direction, orientation):
    native = _normalize(native_direction)
    if native is None or not orientation.valid or orientation.rotation is None:
        return None
    return _normalize(orientation.rotation @ native)


def _angle_2d(first, second):
    a = np.asarray(first, dtype=float); b = np.asarray(second, dtype=float)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9 or not np.all(np.isfinite([*a, *b])):
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def projection_agreement(direction_camera, image_landmarks_px, intrinsics):
    """GT-free agreement with visible MCP->TIP and PIP->TIP axes.

    All axes are compared in undistorted normalized image coordinates. The
    projected 3D tangent is the perspective derivative at the observed TIP.
    """
    direction = _normalize(direction_camera)
    image = np.asarray(image_landmarks_px, dtype=np.float64)
    if direction is None or image.shape not in ((21, 2), (21, 3)):
        return {"mcp_tip_deg": None, "pip_tip_deg": None}
    selected = np.ascontiguousarray(image[[5, 6, 8], :2].reshape(-1, 1, 2))
    undistorted = cv2.undistortPoints(
        selected, camera_matrix(intrinsics), distortion_coefficients(intrinsics)
    ).reshape(-1, 2)
    mcp, pip, tip = undistorted
    dx, dy, dz = direction
    tangent = np.asarray([dx - tip[0] * dz, dy - tip[1] * dz], dtype=float)
    return {
        "mcp_tip_deg": _angle_2d(tangent, tip - mcp),
        "pip_tip_deg": _angle_2d(tangent, tip - pip),
    }


def solve_orientation_hypotheses(world_landmarks, image_landmarks_px, intrinsics,
                                 config, native_directions):
    estimates = {
        R21_CURRENT: solve_current(world_landmarks, image_landmarks_px, intrinsics,
                                  config["current"]),
        R21_RANSAC: solve_ransac21(world_landmarks, image_landmarks_px, intrinsics,
                                  config["ransac"]),
        R_PALM_STABLE: solve_palm_stable(world_landmarks, image_landmarks_px,
                                        intrinsics, config["palm_stable"]),
        RAW_WORLD_DIAGNOSTIC: raw_world_orientation(),
    }
    output = {}
    for name, estimate in estimates.items():
        agreements = {}
        for candidate_name, native in native_directions.items():
            camera_direction = direction_to_camera(native, estimate)
            agreements[candidate_name] = projection_agreement(
                camera_direction, image_landmarks_px, intrinsics
            )
        output[name] = OrientationEstimate(
            **{**estimate.__dict__, "projection_agreement": agreements}
        )
    return output


class OrientationSelector:
    """GT-free production selection from internal geometry only."""

    def __init__(self, config):
        self.config = config
        self.previous = {}

    def reset(self):
        self.previous.clear()

    def _acceptable(self, estimate, candidate_name):
        if not estimate.valid or estimate.rotation is None:
            return False
        values = estimate.projection_agreement.get(candidate_name, {})
        finite = [float(v) for v in values.values() if v is not None and np.isfinite(v)]
        if not finite or float(np.mean(finite)) > float(self.config["max_projection_agreement_deg"]):
            return False
        previous = self.previous.get(estimate.name)
        jump = rotation_geodesic_deg(previous, estimate.rotation)
        if jump is not None and jump > float(self.config["max_rotation_jump_deg"]):
            return False
        return True

    def select(self, estimates, candidate_name="C"):
        for name in self.config["priority"]:
            estimate = estimates.get(name)
            if estimate is not None and self._acceptable(estimate, candidate_name):
                self.previous[name] = estimate.rotation.copy()
                return name, estimate, "GOOD"
        return "ORIENTATION_UNRELIABLE", None, "UNRELIABLE"
