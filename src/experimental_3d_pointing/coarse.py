from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.experimental_3d_pointing.intrinsics import (
    camera_matrix,
    distortion_coefficients,
)


@dataclass(frozen=True)
class NormalizedHit:
    valid: bool
    status: str
    camera_xy: np.ndarray | None = None
    eval_xy: np.ndarray | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    region: str | None = None


def _normalize(direction):
    value = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(value)) if value.shape == (3,) else 0.0
    return value / norm if norm > 1e-9 and np.all(np.isfinite(value)) else None


def undistort_tip_pixel(tip_pixel, intrinsics):
    point = np.asarray(tip_pixel, dtype=np.float64).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        raise ValueError("TIP pixel must be finite")
    source = np.ascontiguousarray(point[:2].reshape(1, 1, 2))
    return cv2.undistortPoints(
        source, camera_matrix(intrinsics), distortion_coefficients(intrinsics)
    ).reshape(2)


def camera_hit_to_eval(hit_camera):
    x, y = np.asarray(hit_camera, dtype=float)
    # A front-facing camera sees the user's physical RIGHT on raw image-left.
    # Eval +X is user physical RIGHT and Eval +Y is user physical UP.
    return np.asarray([-x, -y], dtype=float)


def map_coarse_region(yaw_deg, pitch_deg, threshold_x_deg=4.0,
                      threshold_y_deg=4.0):
    yaw = float(yaw_deg); pitch = float(pitch_deg)
    if not np.all(np.isfinite([yaw, pitch])):
        return None
    horizontal = "CENTER" if abs(yaw) <= float(threshold_x_deg) else (
        "RIGHT" if yaw > 0 else "LEFT"
    )
    vertical = "CENTER" if abs(pitch) <= float(threshold_y_deg) else (
        "UP" if pitch > 0 else "DOWN"
    )
    if horizontal == "CENTER":
        return vertical
    if vertical == "CENTER":
        return horizontal
    return f"{horizontal}_{vertical}"


def normalized_camera_plane_hit(tip_pixel, direction_camera, intrinsics,
                                threshold_x_deg=4.0, threshold_y_deg=4.0,
                                epsilon=1e-6, **_ignored):
    """Depth-free infinite camera-plane hit; extra kwargs are intentionally ignored.

    The function only consumes the raw TIP pixel, calibrated intrinsics and a
    Camera-space direction. Scene depth, anchor position, A4 size and GT are not
    parameters of the computation.
    """
    direction = _normalize(direction_camera)
    if direction is None:
        return NormalizedHit(False, "INVALID_DIRECTION")
    dx, dy, dz = direction
    if dz >= -float(epsilon):
        return NormalizedHit(False, "AWAY_OR_NEAR_PARALLEL")
    try:
        xn, yn = undistort_tip_pixel(tip_pixel, intrinsics)
    except (ValueError, cv2.error):
        return NormalizedHit(False, "INVALID_TIP_PIXEL")
    camera_xy = np.asarray([xn - dx / dz, yn - dy / dz], dtype=float)
    if not np.all(np.isfinite(camera_xy)):
        return NormalizedHit(False, "NON_FINITE_HIT")
    eval_xy = camera_hit_to_eval(camera_xy)
    yaw = float(np.degrees(np.arctan(eval_xy[0])))
    pitch = float(np.degrees(np.arctan(eval_xy[1])))
    region = map_coarse_region(yaw, pitch, threshold_x_deg, threshold_y_deg)
    return NormalizedHit(True, "VALID", camera_xy, eval_xy, yaw, pitch, region)


def select_candidate_no_gt(candidate_c, candidate_b, c_quality, b_quality,
                           c_axis_valid, accepted_quality=("GOOD", "MARGINAL")):
    accepted = set(accepted_quality)
    if candidate_c is not None and candidate_c.valid and c_axis_valid and c_quality in accepted:
        return "C", candidate_c
    if candidate_b is not None and candidate_b.valid and b_quality in accepted:
        return "B_FALLBACK", candidate_b
    return "INVALID", None


def trial_coarse_median(hits, threshold_x_deg=4.0, threshold_y_deg=4.0):
    valid = [hit for hit in hits if hit is not None and hit.valid]
    if not valid:
        return NormalizedHit(False, "NO_VALID_FRAMES")
    yaw = float(np.median([hit.yaw_deg for hit in valid]))
    pitch = float(np.median([hit.pitch_deg for hit in valid]))
    eval_xy = np.asarray([np.tan(np.radians(yaw)), np.tan(np.radians(pitch))])
    camera_xy = np.asarray([-eval_xy[0], -eval_xy[1]])
    region = map_coarse_region(yaw, pitch, threshold_x_deg, threshold_y_deg)
    return NormalizedHit(True, "VALID", camera_xy, eval_xy, yaw, pitch, region)


class CoarseTemporalStabilizer:
    def __init__(self, window_size=7, min_stable_frames=4, hold_frames=6,
                 max_jitter_deg=3.5, threshold_x_deg=4.0,
                 threshold_y_deg=4.0):
        self.window = deque(maxlen=int(window_size))
        self.min_stable = int(min_stable_frames)
        self.hold_frames = int(hold_frames)
        self.max_jitter = float(max_jitter_deg)
        self.tx = float(threshold_x_deg); self.ty = float(threshold_y_deg)
        self.last_region = None; self.remaining_hold = 0

    def reset(self):
        self.window.clear(); self.last_region = None; self.remaining_hold = 0

    def update(self, hit, pointing=True):
        if pointing and hit is not None and hit.valid:
            self.window.append((float(hit.yaw_deg), float(hit.pitch_deg)))
        else:
            self.window.clear()
        if len(self.window) >= self.min_stable:
            values = np.asarray(self.window, dtype=float)
            center = np.median(values, axis=0)
            jitter = float(np.max(np.std(values, axis=0)))
            if jitter <= self.max_jitter:
                self.last_region = map_coarse_region(center[0], center[1], self.tx, self.ty)
                self.remaining_hold = self.hold_frames
                return "STABLE_POINTING", self.last_region, center, jitter
        if self.last_region is not None and self.remaining_hold > 0:
            self.remaining_hold -= 1
            return "HOLDING", self.last_region, None, None
        self.last_region = None
        return "NO_STABLE_POINTING", None, None, None
