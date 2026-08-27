from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.coordinate_adapter import camera_hit_to_eval_mm, nearest_region
from evaluation.diagnostics import assess_direction
from evaluation.z0_geometry import (diagnostic_directions, intersect_ray_with_z0,
                                    validate_hit_range)
from src.camera.calibration import load_intrinsics
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_estimator import DepthEstimator
from src.depth.depth_sampler import sample_point
from src.geometry.backprojection import pixel_depth_to_camera_xyz
from src.geometry.pointing_ray import make_ray
from src.hand.hand_landmarker import HandLandmarker


@dataclass
class EvalFrameResult:
    timestamp: str
    hand_detected: bool
    gesture_label: str
    gesture_confidence: float
    landmarks_2d: np.ndarray | None = None
    direction_2d: np.ndarray | None = None
    points_2d: dict[str, np.ndarray | None] = field(default_factory=dict)
    points_3d: dict[str, np.ndarray | None] = field(default_factory=dict)
    baseline_direction: np.ndarray | None = None
    raw_direction: np.ndarray | None = None
    direction_norm: float | None = None
    angle_to_camera_axis_deg: float | None = None
    direction_quality: str = "INVALID"
    ray_status: str = "INVALID"
    depth_order_status: str = "MISSING_KEYPOINT"
    sanity_flags: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    ray_valid: bool = False
    intersection_valid: bool = False
    intersection_status: str = "NO_HAND"
    raw_hit_eval_mm: np.ndarray | None = None
    hit_eval_mm: np.ndarray | None = None
    pred_region: str | None = None
    depth_mode: str = "UNAVAILABLE"
    intrinsics_mode: str = "UNKNOWN"
    fps: float = 0.0


class Z0EvaluationPipeline:
    """Evaluation-only wrapper around existing hand/depth/geometry interfaces."""
    LANDMARKS = {"mcp": 5, "pip": 6, "dip": 7, "tip": 8}

    def __init__(self, project_config, eval_config, project_root, frame_size, targets):
        self.project_config = project_config
        width, height = frame_size
        self.intrinsics = load_intrinsics(
            project_root / project_config["camera"]["intrinsics"], width, height)
        self.hand = HandLandmarker(**project_config["hand"])
        dcfg = project_config["depth"].copy()
        dcfg["backend"] = eval_config["depth"]["backend"]
        self.depth = DepthEstimator(dcfg["model_id"], dcfg["backend"], dcfg["fallback_depth_m"])
        self.depth_cal = DepthCalibration.load(project_root / dcfg["calibration_file"])
        self.patch_size = dcfg["patch_size"]
        self.depth_alpha = float(dcfg["ema_alpha"])
        self.depth_ema = {}
        self.targets = targets
        self.epsilon = float(eval_config["intersection_epsilon"])
        self.quality_config = eval_config["direction_quality"]
        self.target_plane = eval_config["target_plane"]

    def _smooth(self, name, value):
        if value is None:
            return None
        previous = self.depth_ema.get(name, value)
        smoothed = self.depth_alpha * value + (1-self.depth_alpha) * previous
        self.depth_ema[name] = smoothed
        return smoothed

    def process(self, frame, timestamp: str, fps=0.0) -> EvalFrameResult:
        hand = self.hand.process(frame)
        mode = self.depth.mode + ("_CORRECTED" if self.depth_cal.calibrated else "_UNCALIBRATED")
        result = EvalFrameResult(timestamp, hand.detected,
                                 "POINTING" if hand.is_pointing else ("HAND" if hand.detected else "NO_HAND"),
                                 hand.confidence, depth_mode=mode,
                                 intrinsics_mode=self.intrinsics.mode, fps=float(fps))
        if not hand.detected:
            return result
        # Visualization-only exposure of the exact same detection result.
        # No second detector or geometry path is introduced.
        result.landmarks_2d = hand.landmarks_2d.copy()
        result.direction_2d = hand.direction_2d.copy()
        result.points_2d = {name: hand.landmarks_2d[index, :2].copy()
                            for name, index in self.LANDMARKS.items()}
        depth_map = self.depth.process(frame)
        for name, point in result.points_2d.items():
            raw = sample_point(depth_map, point, self.patch_size)
            corrected = self.depth_cal.correct(raw) if raw is not None else None
            depth = self._smooth(name, corrected)
            result.points_3d[name] = (pixel_depth_to_camera_xyz(*point, depth, self.intrinsics)
                                      if depth is not None else None)
        result.diagnostics = diagnostic_directions(**{
            f"{name}_xyz": result.points_3d.get(name) for name in self.LANDMARKS})
        pip, tip = result.points_3d.get("pip"), result.points_3d.get("tip")
        if pip is None or tip is None:
            result.intersection_status = "MISSING_3D_KEYPOINT"
            return result
        try:
            origin, direction = make_ray(pip, tip)  # Existing business baseline.
            result.ray_valid = True
            result.baseline_direction = direction
        except ValueError:
            result.intersection_status = "INVALID_DIRECTION"
            return result
        assessment=assess_direction(pip,tip,direction,self.quality_config)
        result.raw_direction=assessment.raw_direction
        result.direction_norm=assessment.direction_norm
        result.angle_to_camera_axis_deg=assessment.angle_to_camera_axis_deg
        result.direction_quality=assessment.quality
        result.ray_status=assessment.ray_status
        result.depth_order_status=assessment.depth_order_status
        result.sanity_flags=assessment.sanity_flags
        if assessment.quality == "INVALID":
            result.ray_valid = False
            result.intersection_status = "INVALID_DIRECTION"
            return result
        near_parallel=max(self.epsilon,float(self.quality_config["near_parallel_abs_dz"]))
        intersection = intersect_ray_with_z0(origin, direction, near_parallel)
        result.intersection_valid = intersection.valid
        result.intersection_status = intersection.status
        if intersection.valid:
            result.raw_hit_eval_mm = camera_hit_to_eval_mm(intersection.point_camera)
            range_status=validate_hit_range(result.raw_hit_eval_mm,
                                            self.target_plane["width_mm"],
                                            self.target_plane["height_mm"])
            if range_status=="VALID":
                result.hit_eval_mm=result.raw_hit_eval_mm.copy()
                result.pred_region=nearest_region(*result.hit_eval_mm,self.targets)
            else:
                result.intersection_valid=False
                result.intersection_status=range_status
        return result

    def close(self):
        self.hand.close()
