from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectRayMetric:
    object_id: int
    forward_m: float
    perpendicular_m: float
    angle_deg: float
    angular_tolerance_deg: float
    score: float


def _normalize(value):
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector)) if vector.shape == (3,) else 0.0
    return vector / norm if norm > 1e-9 and np.all(np.isfinite(vector)) else None


def bbox_angular_radius_deg(bbox, intrinsics):
    x1, y1, x2, y2 = bbox
    half_width = max(0.0, float(x2 - x1) / 2.0)
    half_height = max(0.0, float(y2 - y1) / 2.0)
    ax = np.arctan2(half_width, float(intrinsics.fx))
    ay = np.arctan2(half_height, float(intrinsics.fy))
    return float(np.degrees(np.hypot(ax, ay)))


def select_object_by_ray(objects, origin_camera, direction_camera, intrinsics,
                         base_max_angle_deg=8.0,
                         max_perpendicular_distance_m=.35,
                         minimum_confidence=.38,
                         angular_weight=.65,
                         perpendicular_weight=.25,
                         confidence_weight=.10,
                         bbox_tolerance_scale=1.0,
                         **_ignored):
    origin = np.asarray(origin_camera, dtype=float)
    direction = _normalize(direction_camera)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)) or direction is None:
        return None, None, []
    candidates = []
    for detected in objects:
        if not detected.depth_valid or detected.center_camera is None \
                or float(detected.confidence) < float(minimum_confidence):
            continue
        vector = np.asarray(detected.center_camera, dtype=float) - origin
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-9 or not np.isfinite(distance): continue
        forward = float(np.dot(vector, direction))
        if forward <= 0: continue
        perpendicular = float(np.linalg.norm(vector - forward * direction))
        angle = float(np.degrees(np.arccos(np.clip(forward / distance, -1.0, 1.0))))
        tolerance = float(base_max_angle_deg) + float(bbox_tolerance_scale) * \
            bbox_angular_radius_deg(detected.bbox, intrinsics)
        if angle > tolerance or perpendicular > float(max_perpendicular_distance_m):
            continue
        score = (float(angular_weight) * angle / max(tolerance, 1e-6) +
                 float(perpendicular_weight) * perpendicular /
                    max(float(max_perpendicular_distance_m), 1e-6) +
                 float(confidence_weight) * (1.0 - float(detected.confidence)))
        candidates.append((score, detected, ObjectRayMetric(
            detected.id, forward, perpendicular, angle, tolerance, score)))
    if not candidates: return None, None, []
    candidates.sort(key=lambda item: item[0])
    _, selected, metric = candidates[0]
    return selected, metric, [item[2] for item in candidates]


class ObjectSelectionHysteresis:
    def __init__(self, switch_confirm_frames=3, release_frames=3):
        self.switch_frames = int(switch_confirm_frames); self.release_frames = int(release_frames)
        self.current = None; self.pending = None; self.pending_count = 0; self.release_count = 0

    def reset(self):
        self.current = None; self.pending = None; self.pending_count = 0; self.release_count = 0

    def update(self, candidate_id):
        if candidate_id == self.current:
            self.pending = None; self.pending_count = 0; self.release_count = 0
        elif candidate_id is None:
            self.release_count += 1
            if self.release_count >= self.release_frames: self.current = None
        else:
            self.release_count = 0
            if candidate_id != self.pending:
                self.pending = candidate_id; self.pending_count = 1
            else: self.pending_count += 1
            if self.pending_count >= self.switch_frames:
                self.current = candidate_id; self.pending = None; self.pending_count = 0
        return self.current
