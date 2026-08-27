from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.direction_candidates import fit_finger_axis_with_quality
from src.experimental_3d_pointing.coarse import (
    CoarseTemporalStabilizer, NormalizedHit, normalized_camera_plane_hit,
)
from src.experimental_3d_pointing.core import (
    AnchorTemporalFilter, DirectionEMA, palm_anchor_candidates,
    reconstruct_tip_camera,
)
from src.experimental_3d_pointing.intrinsics import backproject_distorted_pixel
from src.experimental_3d_pointing.orientation import (
    OrientationSelector, direction_to_camera, solve_orientation_hypotheses,
)


def _normalize(value):
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector)) if vector.shape == (3,) else 0.0
    return vector / norm if norm > 1e-9 and np.all(np.isfinite(vector)) else None


class PointingGestureHysteresis:
    def __init__(self, confirm_frames=2, hold_frames=3):
        self.confirm_frames = int(confirm_frames); self.hold_frames = int(hold_frames)
        self.active = False; self.confirm_count = 0; self.missing_count = 0

    def reset(self):
        self.active = False; self.confirm_count = 0; self.missing_count = 0

    def update(self, raw_pointing):
        if raw_pointing:
            self.confirm_count += 1; self.missing_count = 0
            if self.confirm_count >= self.confirm_frames: self.active = True
        else:
            self.confirm_count = 0
            if self.active:
                self.missing_count += 1
                if self.missing_count > self.hold_frames: self.active = False
        return self.active


@dataclass
class UnifiedHandState:
    detected: bool = False
    normalized_landmarks: np.ndarray | None = None
    pixel_landmarks: np.ndarray | None = None
    world_landmarks_m: np.ndarray | None = None
    handedness: str | None = None
    handedness_score: float | None = None
    is_pointing: bool = False
    gesture_confidence: float = 0.0

    @classmethod
    def from_probe(cls, probe):
        return cls(probe.detected, probe.normalized_landmarks, probe.pixel_landmarks,
                   probe.world_landmarks_m, probe.handedness, probe.handedness_score,
                   probe.is_pointing, probe.gesture_confidence)


@dataclass
class UnifiedRayResult:
    status: str = "NO_HAND"
    gesture_active: bool = False
    orientation_name: str = "ORIENTATION_UNRELIABLE"
    orientation_quality: str = "UNRELIABLE"
    direction_source: str = "INVALID"
    origin_camera: np.ndarray | None = None
    direction_camera: np.ndarray | None = None
    finger_distance_m: float | None = None
    anchor_camera: np.ndarray | None = None
    anchor_depth_m: float | None = None
    anchor_status: str = "INVALID"
    coarse_hit: NormalizedHit | None = None
    stable_region: str | None = None
    region_temporal_status: str = "NO_STABLE_POINTING"
    candidate_b_camera: np.ndarray | None = None
    candidate_c_camera: np.ndarray | None = None
    axis_residual_m: float | None = None
    axis_linearity: float | None = None
    pnp_rmse_px: float | None = None
    selected_projection_agreement: dict = field(default_factory=dict)
    orientation_hypotheses: dict = field(default_factory=dict)


class UnifiedFingerRayCore:
    """One Camera-space ray shared by object selection and coarse region."""

    def __init__(self, config):
        self.config = config
        self.orientation_selector = OrientationSelector(config["orientation"]["selection"])
        direction_cfg = config["direction_temporal"]
        self.direction_ema = DirectionEMA(direction_cfg["enabled"],
            direction_cfg["ema_alpha"], direction_cfg["max_angle_jump_deg"])
        anchor_temporal = config["anchor_temporal"]
        self.anchor_filter = AnchorTemporalFilter(anchor_temporal["enabled"],
            anchor_temporal["ema_alpha"], anchor_temporal["max_jump_m"],
            anchor_temporal["max_invalid_frames"])
        gesture = config["gesture_temporal"]
        self.gesture = PointingGestureHysteresis(gesture["confirm_frames"], gesture["hold_frames"])
        coarse = config["coarse_region"]
        self.region_temporal = CoarseTemporalStabilizer(
            coarse["window_size"], coarse["min_stable_frames"], coarse["hold_frames"],
            coarse["max_jitter_deg"], coarse["center_half_angle_x_deg"],
            coarse["center_half_angle_y_deg"])

    def reset(self):
        self.orientation_selector.reset(); self.direction_ema.reset(); self.anchor_filter.reset()
        self.gesture.reset(); self.region_temporal.reset()

    def process(self, probe, depth_map, intrinsics, depth_corrector=None):
        output = UnifiedRayResult()
        if not probe.detected:
            self.gesture.update(False); self.region_temporal.update(None, False)
            return output
        output.status = "HAND_DETECTED"
        gesture_active = self.gesture.update(bool(probe.is_pointing))
        output.gesture_active = gesture_active
        if probe.world_landmarks_m is None or probe.pixel_landmarks is None:
            output.status = "NOT_POINTING" if not gesture_active else "POINTING_UNSTABLE"
            return output
        world = np.asarray(probe.world_landmarks_m, dtype=float)
        pixels = np.asarray(probe.pixel_landmarks, dtype=float)
        b_native = _normalize(world[8] - world[6])
        axis = fit_finger_axis_with_quality(world)
        c_native = axis.direction if axis.valid else None
        output.axis_residual_m = axis.residual_m; output.axis_linearity = axis.linearity
        estimates = solve_orientation_hypotheses(
            world, pixels, intrinsics, self.config["orientation"],
            {"B": b_native, "C": c_native})
        output.orientation_hypotheses = estimates
        axis_config = self.config["axis"]
        axis_good = bool(axis.valid and axis.residual_m <= float(axis_config["max_residual_m"])
            and axis.linearity >= float(axis_config["minimum_linearity"]))
        orientation_candidate = "C" if axis_good else "B"
        orientation_name, orientation, orientation_quality = self.orientation_selector.select(
            estimates, orientation_candidate)
        output.orientation_name = orientation_name; output.orientation_quality = orientation_quality
        if orientation is None:
            output.status = "NOT_POINTING" if not gesture_active else "POINTING_UNSTABLE"
            self.region_temporal.update(None, False)
            return output
        output.pnp_rmse_px = orientation.reprojection_rmse_px
        output.selected_projection_agreement = orientation.projection_agreement
        b_camera = direction_to_camera(b_native, orientation)
        c_camera = direction_to_camera(c_native, orientation)
        output.candidate_b_camera = b_camera; output.candidate_c_camera = c_camera
        raw_direction = c_camera if axis_good and c_camera is not None else b_camera
        output.direction_source = "C" if raw_direction is c_camera else "B_FALLBACK"
        smoothed, _, jump_exceeded = self.direction_ema.update("UNIFIED", raw_direction)
        output.direction_camera = smoothed
        if smoothed is None or jump_exceeded:
            output.status = "NOT_POINTING" if not gesture_active else "POINTING_UNSTABLE"
            self.region_temporal.update(None, False)
            return output

        anchor_cfg = self.config["anchor"]
        _, anchor = palm_anchor_candidates(depth_map, pixels,
            anchor_cfg["sample_landmark_indices"], anchor_cfg["patch_size"],
            anchor_cfg["minimum_valid_samples"], anchor_cfg["max_depth_mad_m"],
            depth_corrector, anchor_cfg["lower_percentile"], anchor_cfg["upper_percentile"],
            anchor_cfg["outlier_mad_scale"])
        output.anchor_depth_m = anchor.median_m; output.anchor_status = anchor.status
        filtered_depth, temporal_anchor_status = self.anchor_filter.update(anchor)
        anchor_index = int(anchor_cfg["landmark_index"])
        if filtered_depth is not None and temporal_anchor_status == "VALID":
            try:
                output.anchor_camera = backproject_distorted_pixel(
                    *pixels[anchor_index, :2], filtered_depth, intrinsics)
                output.origin_camera = reconstruct_tip_camera(
                    output.anchor_camera, orientation.rotation, world[8], world[anchor_index])
                output.finger_distance_m = float(np.linalg.norm(output.origin_camera))
            except ValueError:
                output.origin_camera = None

        coarse_cfg = self.config["coarse_region"]
        output.coarse_hit = normalized_camera_plane_hit(
            pixels[8, :2], smoothed, intrinsics,
            coarse_cfg["center_half_angle_x_deg"], coarse_cfg["center_half_angle_y_deg"],
            coarse_cfg["intersection_epsilon"])
        temporal_status, stable_region, _, _ = self.region_temporal.update(
            output.coarse_hit, gesture_active)
        output.region_temporal_status = temporal_status; output.stable_region = stable_region
        if not gesture_active: output.status = "NOT_POINTING"
        elif orientation_quality != "GOOD": output.status = "POINTING_UNSTABLE"
        else: output.status = "POINTING_VALID"
        return output
