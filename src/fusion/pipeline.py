from __future__ import annotations

import numpy as np

from src.camera.calibration import load_intrinsics
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_sampler import sample_bbox, sample_point
from src.geometry.backprojection import pixel_depth_to_camera_xyz
from src.geometry.pointing_ray import make_ray
from src.geometry.target_selector import HysteresisSelector, select_target
from src.hand.hand_landmarker import HandLandmarker
from src.types import HandState, PipelineResult


class VisionPipeline:
    def __init__(self, config, project_root, frame_size):
        self.cfg, self.root = config, project_root
        w, h = frame_size
        self.intrinsics = load_intrinsics(project_root / config["camera"]["intrinsics"], w, h)
        hand_cfg = config["hand"]
        self.hand = HandLandmarker(**hand_cfg)
        self.detector = None
        self.depth = None
        self.init_errors = {}
        try:
            from src.detection.cup_detector import CupDetector
            self.detector = CupDetector(**config["detection"])
        except Exception as exc:
            self.init_errors["detector"] = f"{type(exc).__name__}: {exc}"
        try:
            from src.depth.depth_estimator import DepthEstimator
            dcfg = config["depth"]
            self.depth = DepthEstimator(dcfg["model_id"], dcfg["backend"], dcfg["fallback_depth_m"])
            if self.depth.error:
                self.init_errors["depth"] = self.depth.error
        except Exception as exc:
            self.init_errors["depth"] = f"{type(exc).__name__}: {exc}"
        self.depth_cal = DepthCalibration.load(project_root / config["depth"]["calibration_file"])
        tcfg = config["target_selection"]
        self.hysteresis = HysteresisSelector(tcfg["switch_confirm_frames"], tcfg["release_frames"])
        self.depth_ema = {}

    def _smooth_depth(self, key, value):
        if value is None:
            return None
        alpha = float(self.cfg["depth"]["ema_alpha"])
        old = self.depth_ema.get(key, value)
        self.depth_ema[key] = alpha*value + (1-alpha)*old
        return self.depth_ema[key]

    def process(self, frame):
        hand = self.hand.process(frame)
        cups = self.detector.process(frame) if self.detector else []
        if self.depth:
            depth_map = self.depth.process(frame)
            depth_mode = self.depth.mode + ("_CORRECTED" if self.depth_cal.calibrated else "_UNCALIBRATED")
        else:
            depth_map, depth_mode = None, "UNAVAILABLE"
        dcfg = self.cfg["depth"]
        tip_3d = pip_3d = ray_direction = None
        raw_tip = raw_pip = None
        if hand.detected and depth_map is not None:
            raw_tip = sample_point(depth_map, hand.index_tip, dcfg["patch_size"])
            raw_pip = sample_point(depth_map, hand.index_pip, dcfg["patch_size"])
            tip_depth = self._smooth_depth("tip", self.depth_cal.correct(raw_tip) if raw_tip else None)
            pip_depth = self._smooth_depth("pip", self.depth_cal.correct(raw_pip) if raw_pip else None)
            if tip_depth and pip_depth:
                tip_3d = pixel_depth_to_camera_xyz(*hand.index_tip, tip_depth, self.intrinsics)
                pip_3d = pixel_depth_to_camera_xyz(*hand.index_pip, pip_depth, self.intrinsics)
                try:
                    _, ray_direction = make_ray(pip_3d, tip_3d)
                except ValueError:
                    pass
        for cup in cups:
            cup.raw_depth = sample_bbox(depth_map, cup.bbox, dcfg["bbox_inner_ratio"])
            corrected = self.depth_cal.correct(cup.raw_depth) if cup.raw_depth else None
            cup.depth = self._smooth_depth(f"cup_{cup.id}", corrected)
            if cup.depth:
                cup.center_3d = pixel_depth_to_camera_xyz(*cup.center_2d, cup.depth, self.intrinsics)
        candidate = score = None
        if hand.detected and hand.is_pointing and np.linalg.norm(hand.direction_2d) > 0:
            candidate, score = select_target(cups, hand.index_tip, hand.direction_2d,
                                             tip_3d, ray_direction, **{
                k: v for k, v in self.cfg["target_selection"].items()
                if k not in ("switch_confirm_frames", "release_frames")})
        selected = self.hysteresis.update(candidate)
        return PipelineResult(hand, cups, selected, score, depth_map, depth_mode,
                              tip_3d, pip_3d, ray_direction,
                              {"raw_tip_depth": raw_tip, "raw_pip_depth": raw_pip,
                               "init_errors": self.init_errors,
                               "calibration": self.intrinsics.mode})

    def close(self):
        self.hand.close()

