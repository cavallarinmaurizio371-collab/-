from __future__ import annotations

import numpy as np


def _vectors(frames, collection, name, dimensions):
    values = []
    for frame in frames:
        value = getattr(frame, collection, {}).get(name)
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.shape == (dimensions,) and np.all(np.isfinite(array)):
                values.append(array)
    return np.stack(values) if values else np.empty((0, dimensions))


def _radial_std(values):
    if len(values) == 0:
        return None
    centered = values - values.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def analyze_stability(frames, thresholds=None) -> dict:
    thresholds = thresholds or {}
    metrics = {}
    unstable_reasons = []
    for name in ("tip", "pip", "dip"):
        points_2d = _vectors(frames, "points_2d", name, 2)
        points_3d = _vectors(frames, "points_3d", name, 3)
        pixel_std = _radial_std(points_2d)
        xyz_std = _radial_std(points_3d)
        metrics[f"{name}_pixel_std"] = pixel_std
        metrics[f"{name}_xyz_std_m"] = xyz_std
        metrics[f"{name}_z_std_m"] = float(np.std(points_3d[:,2])) if len(points_3d) else None
        metrics[f"{name}_valid_count"] = int(len(points_3d))
        if pixel_std is not None and pixel_std > thresholds.get("keypoint_pixel_std_threshold", float("inf")):
            unstable_reasons.append(f"{name.upper()}_2D")
        if xyz_std is not None and xyz_std > thresholds.get("keypoint_xyz_std_m_threshold", float("inf")):
            unstable_reasons.append(f"{name.upper()}_3D")
    directions = [np.asarray(frame.baseline_direction, dtype=float) for frame in frames
                  if frame.baseline_direction is not None]
    if directions:
        directions = np.stack([d/np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 1e-9])
    if len(directions):
        mean_direction = directions.mean(axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        angles = np.degrees(np.arccos(np.clip(directions @ mean_direction, -1.0, 1.0)))
        direction_std = float(np.std(angles))
        dz_negative_rate = float(np.mean(directions[:, 2] < 0))
    else:
        direction_std = dz_negative_rate = None
    metrics["direction_angle_std_deg"] = direction_std
    metrics["direction_dz_negative_rate"] = dz_negative_rate
    if direction_std is not None and direction_std > thresholds.get("direction_angle_std_deg_threshold", float("inf")):
        unstable_reasons.append("DIRECTION")
    hits = np.stack([frame.hit_eval_mm for frame in frames if frame.hit_eval_mm is not None]) \
        if any(frame.hit_eval_mm is not None for frame in frames) else np.empty((0, 2))
    metrics["hit_x_std_mm"] = float(np.std(hits[:, 0])) if len(hits) else None
    metrics["hit_y_std_mm"] = float(np.std(hits[:, 1])) if len(hits) else None
    metrics["hit_radial_std_mm"] = _radial_std(hits)
    ordering=[]
    for frame in frames:
        tip=frame.points_3d.get("tip") if frame.points_3d else None
        pip=frame.points_3d.get("pip") if frame.points_3d else None
        if tip is not None and pip is not None:
            ordering.append(float(tip[2])<float(pip[2]))
    metrics["tip_pip_depth_order_consistency"] = float(np.mean(ordering)) if ordering else None
    if metrics["hit_radial_std_mm"] is not None and metrics["hit_radial_std_mm"] > thresholds.get("hit_radial_std_mm_threshold", float("inf")):
        unstable_reasons.append("HIT_POINT")
    metrics["stability_status"] = "KEYPOINT_UNSTABLE" if unstable_reasons else "STABLE"
    metrics["unstable_reasons"] = unstable_reasons
    return metrics
