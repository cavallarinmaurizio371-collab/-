from __future__ import annotations

from collections import defaultdict

import numpy as np

from evaluation.coordinate_adapter import REGION_ORDER


def _finite(values):
    return [float(value) for value in values if value not in (None, "") and np.isfinite(float(value))]


def error_statistics(trials) -> dict:
    errors = _finite(trial.get("radial_error_mm") for trial in trials)
    return {
        "mean_radial_error_mm": float(np.mean(errors)) if errors else None,
        "median_radial_error_mm": float(np.median(errors)) if errors else None,
        "p90_radial_error_mm": float(np.percentile(errors, 90)) if errors else None,
        "max_radial_error_mm": float(np.max(errors)) if errors else None,
    }


def _rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _frame_rates(frames):
    total = len(frames)
    detected = sum(bool(row.get("hand_detected")) for row in frames)
    pointing = sum(row.get("gesture_label") == "POINTING" for row in frames)
    directions = sum(bool(row.get("ray_valid")) for row in frames)
    intersections = sum(bool(row.get("intersection_valid")) for row in frames)
    return {
        "hand_detection_rate": _rate(detected, total),
        "pointing_recognition_rate": _rate(pointing, detected),
        "valid_3d_direction_rate": _rate(directions, detected),
        "valid_z0_intersection_rate": _rate(intersections, detected),
    }


def _trial_group(items):
    detected = sum(int(item.get("hand_detected_frames") or 0) for item in items)
    direction = sum(int(item.get("valid_direction_frames") or 0) for item in items)
    intersection = sum(int(item.get("valid_intersection_frames") or 0) for item in items)
    valid = [item for item in items if item.get("pred_region")]
    return {
        "trial_count": len(items),
        "direction_valid_rate": _rate(direction, detected),
        "intersection_rate": _rate(intersection, detected),
        "region_accuracy": _rate(sum(bool(x.get("region_correct")) for x in valid), len(valid)),
        **error_statistics(items),
    }


def _grouped(trials, field):
    groups = defaultdict(list)
    for trial in trials:
        groups[str(trial.get(field))].append(trial)
    return {key: _trial_group(items) for key, items in sorted(groups.items())}


def confusion_matrix(trials):
    matrix = {gt: {pred: 0 for pred in REGION_ORDER} for gt in REGION_ORDER}
    for trial in trials:
        gt, pred = trial.get("gt_target"), trial.get("pred_region")
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def build_summary(trials, frames, mirror_check="NOT_RUN") -> dict:
    valid = [trial for trial in trials if trial.get("intersection_valid")]
    predicted = [trial for trial in trials if trial.get("pred_region")]
    failures = defaultdict(int)
    for trial in trials:
        failures[str(trial.get("failure_reason") or "UNKNOWN")] += 1
    center_trials = [trial for trial in trials if trial.get("gt_target") == "CENTER"]
    return {
        "total_trials": len(trials),
        "valid_trials": len(valid),
        **_frame_rates(frames),
        "region_accuracy": _rate(sum(bool(x.get("region_correct")) for x in predicted), len(predicted)),
        **error_statistics(trials),
        "tip_pixel_std_mean": float(np.mean(v)) if (v := _finite(t.get("tip_pixel_std") for t in trials)) else None,
        "pip_pixel_std_mean": float(np.mean(v)) if (v := _finite(t.get("pip_pixel_std") for t in trials)) else None,
        "dip_pixel_std_mean": float(np.mean(v)) if (v := _finite(t.get("dip_pixel_std") for t in trials)) else None,
        "direction_jitter_deg_mean": float(np.mean(v)) if (v := _finite(t.get("direction_angle_std_deg") for t in trials)) else None,
        "hit_point_jitter_mm_mean": float(np.mean(v)) if (v := _finite(t.get("hit_radial_std_mm") for t in trials)) else None,
        "by_distance_cm": _grouped(trials, "measured_hand_distance_cm"),
        "by_gt_target": _grouped(trials, "gt_target"),
        "by_hand_position": _grouped(trials, "hand_position_in_frame"),
        "center_gt_by_hand_position": _grouped(center_trials, "hand_position_in_frame"),
        "failure_distribution": {
            key: {"count": count, "rate": _rate(count, len(trials))}
            for key, count in sorted(failures.items())},
        "mirror_check": mirror_check,
    }
