from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.direction_candidates import fit_finger_axis_with_quality
from src.experimental_3d_pointing.coarse import (
    normalized_camera_plane_hit, select_candidate_no_gt, trial_coarse_median,
)
from src.experimental_3d_pointing.intrinsics import load_phase2b_intrinsics
from src.experimental_3d_pointing.orientation import (
    RAW_WORLD_DIAGNOSTIC, R21_CURRENT, R21_RANSAC, R_PALM_STABLE,
    OrientationSelector, direction_to_camera, rotation_geodesic_deg,
    solve_orientation_hypotheses,
)
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path


HYPOTHESES = (R21_CURRENT, R21_RANSAC, R_PALM_STABLE, RAW_WORLD_DIAGNOSTIC)


def _angle(first, second):
    first = np.asarray(first, dtype=float); second = np.asarray(second, dtype=float)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) /
        (np.linalg.norm(first) * np.linalg.norm(second)), -1.0, 1.0))))


def _median(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(values)) if values else None


def analyze(session_dir):
    config = load_yaml(PROJECT_ROOT / "configs/mentor_coarse_pointing.yaml")
    width = int(config["intrinsics"]["required_width"]); height = int(config["intrinsics"]["required_height"])
    intrinsics = load_phase2b_intrinsics(PROJECT_ROOT, {"intrinsics": {
        "calibrated_file": config["intrinsics"]["calibrated_file"]}}, width, height)
    region = config["coarse_region"]
    frames = defaultdict(list)
    for path in sorted(Path(session_dir).glob("*_frames.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line); frames[row["target"]].append(row)
    orientation_samples = {name: [] for name in HYPOTHESES}
    trial_results = []
    for target, rows in frames.items():
        hits = {(name, candidate): [] for name in HYPOTHESES for candidate in ("B", "C")}
        production_hits = []; selector = OrientationSelector(config["orientation"]["selection"])
        for row in rows:
            if row["world_landmarks_m"] is None or row["image_landmarks_px"] is None:
                continue
            world = np.asarray(row["world_landmarks_m"], dtype=float)
            pixels = np.asarray(row["image_landmarks_px"], dtype=float)
            axis = fit_finger_axis_with_quality(world)
            native = {"B": world[8] - world[6], "C": axis.direction if axis.valid else None}
            estimates = solve_orientation_hypotheses(world, pixels, intrinsics,
                                                     config["orientation"], native)
            orientation_candidate = "C" if axis.valid else "B"
            _, selected_orientation, _ = selector.select(estimates, orientation_candidate)
            if selected_orientation is not None and row["pointing"]:
                directions = {name: direction_to_camera(value, selected_orientation)
                              for name, value in native.items()}
                candidate_hits = {name: normalized_camera_plane_hit(
                    row["tip_pixel_raw"], direction, intrinsics,
                    region["center_half_angle_x_deg"], region["center_half_angle_y_deg"],
                    region["intersection_epsilon"])
                    for name, direction in directions.items()}
                axis_valid = bool(axis.valid and
                    axis.linearity >= float(config["candidate"]["c_min_linearity"]) and
                    axis.residual_m <= float(config["candidate"]["c_max_residual_m"]))
                _, selected = select_candidate_no_gt(
                    candidate_hits["C"], candidate_hits["B"],
                    "GOOD" if axis_valid else "HAND_AXIS_UNSTABLE",
                    "GOOD" if directions["B"] is not None else "INVALID",
                    axis_valid, tuple(config["candidate"]["accepted_quality"]),
                )
                if selected is not None and selected.valid:
                    production_hits.append(selected)
            for name, estimate in estimates.items():
                sample = {"target": target, "pointing": bool(row["pointing"]),
                          "valid": estimate.valid, "rmse": estimate.reprojection_rmse_px,
                          "inlier_rate": estimate.inlier_rate,
                          "rotation": estimate.rotation,
                          "rotation_magnitude": estimate.rotation_magnitude_deg}
                for candidate in ("B", "C"):
                    agreement = estimate.projection_agreement.get(candidate, {})
                    sample[f"{candidate}_agreement"] = _median(agreement.values())
                    direction = direction_to_camera(native[candidate], estimate)
                    sample[f"{candidate}_direction"] = direction
                    sample[f"{candidate}_toward"] = bool(direction is not None and direction[2] < 0)
                    if row["pointing"] and direction is not None:
                        hit = normalized_camera_plane_hit(
                            row["tip_pixel_raw"], direction, intrinsics,
                            region["center_half_angle_x_deg"], region["center_half_angle_y_deg"],
                            region["intersection_epsilon"])
                        if hit.valid:
                            hits[(name, candidate)].append(hit)
                orientation_samples[name].append(sample)
        trial = {"target": target, "frame_count": len(rows),
                 "pointing_rate": sum(bool(row["pointing"]) for row in rows) / len(rows)}
        production = trial_coarse_median(production_hits,
            region["center_half_angle_x_deg"], region["center_half_angle_y_deg"])
        trial["PRODUCTION"] = {"valid_frames": len(production_hits),
            "valid": production.valid, "region": production.region,
            "yaw_deg": production.yaw_deg, "pitch_deg": production.pitch_deg}
        for name in HYPOTHESES:
            for candidate in ("B", "C"):
                result = trial_coarse_median(hits[(name, candidate)],
                    region["center_half_angle_x_deg"], region["center_half_angle_y_deg"])
                trial[f"{name}_{candidate}"] = {
                    "valid_frames": len(hits[(name, candidate)]),
                    "valid": result.valid, "region": result.region,
                    "yaw_deg": result.yaw_deg, "pitch_deg": result.pitch_deg,
                }
        trial_results.append(trial)
    orientation_metrics = {}
    for name, samples in orientation_samples.items():
        valid = [sample for sample in samples if sample["valid"]]
        rotation_jumps = []
        direction_jumps = {"B": [], "C": []}
        previous_rotation = None; previous_direction = {"B": None, "C": None}
        for sample in samples:
            if sample["valid"]:
                jump = rotation_geodesic_deg(previous_rotation, sample["rotation"])
                if jump is not None: rotation_jumps.append(jump)
                previous_rotation = sample["rotation"]
                for candidate in ("B", "C"):
                    direction = sample[f"{candidate}_direction"]
                    if direction is not None and previous_direction[candidate] is not None:
                        direction_jumps[candidate].append(_angle(previous_direction[candidate], direction))
                    if direction is not None: previous_direction[candidate] = direction
        orientation_metrics[name] = {
            "valid_rate": len(valid) / len(samples) if samples else 0.0,
            "valid_frames": len(valid), "total_frames": len(samples),
            "rmse_median_px": _median(sample["rmse"] for sample in valid),
            "inlier_rate_median": _median(sample["inlier_rate"] for sample in valid),
            "rotation_magnitude_median_deg": _median(sample["rotation_magnitude"] for sample in valid),
            "rotation_geodesic_jitter_median_deg": _median(rotation_jumps),
            "B_projection_agreement_median_deg": _median(sample["B_agreement"] for sample in valid),
            "C_projection_agreement_median_deg": _median(sample["C_agreement"] for sample in valid),
            "B_toward_rate": (sum(sample["B_toward"] for sample in valid) / len(valid)) if valid else 0.0,
            "C_toward_rate": (sum(sample["C_toward"] for sample in valid) / len(valid)) if valid else 0.0,
            "B_direction_jitter_median_deg": _median(direction_jumps["B"]),
            "C_direction_jitter_median_deg": _median(direction_jumps["C"]),
        }
    # Diagnostic flags are derived without target labels.
    rotated = [orientation_metrics[name]["C_projection_agreement_median_deg"]
               for name in (R21_CURRENT, R21_RANSAC, R_PALM_STABLE)
               if orientation_metrics[name]["C_projection_agreement_median_deg"] is not None]
    raw_agreement = orientation_metrics[RAW_WORLD_DIAGNOSTIC]["C_projection_agreement_median_deg"]
    possible_double = bool(rotated and raw_agreement is not None and raw_agreement + 5.0 < min(rotated))
    valid_production = [trial for trial in trial_results if trial["PRODUCTION"]["valid"]]
    correct_production = [trial for trial in valid_production
                          if trial["PRODUCTION"]["region"] == trial["target"]]
    horizontal_inversions = sum(
        (trial["target"] == "LEFT" and trial["PRODUCTION"]["region"] in ("RIGHT", "RIGHT_UP", "RIGHT_DOWN")) or
        (trial["target"] == "RIGHT" and trial["PRODUCTION"]["region"] in ("LEFT", "LEFT_UP", "LEFT_DOWN"))
        for trial in valid_production)
    vertical_inversions = sum(
        (trial["target"] == "UP" and trial["PRODUCTION"]["region"] in ("DOWN", "LEFT_DOWN", "RIGHT_DOWN")) or
        (trial["target"] == "DOWN" and trial["PRODUCTION"]["region"] in ("UP", "LEFT_UP", "RIGHT_UP"))
        for trial in valid_production)
    return {
        "session": Path(session_dir).name,
        "orientation_metrics": orientation_metrics,
        "trial_results": trial_results,
        "diagnostic_flags": {
            "possible_double_rotation": possible_double,
            "frontal_pnp_degradation": orientation_metrics[R21_CURRENT]["valid_rate"] < .5,
        },
        "production_summary": {
            "valid_trials": len(valid_production), "total_trials": len(trial_results),
            "trial_valid_rate": len(valid_production) / len(trial_results) if trial_results else 0.0,
            "accuracy": len(correct_production) / len(valid_production) if valid_production else None,
            "left_right_inversions": int(horizontal_inversions),
            "up_down_inversions": int(vertical_inversions),
        },
        "accuracy_is_evaluation_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    args = parser.parse_args()
    report = analyze(Path(args.session_dir))
    output = assert_safe_path(Path(args.session_dir) / "orientation_analysis.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                 default=lambda value: value.tolist()), encoding="utf-8")
    print(output)
    print(json.dumps(report["orientation_metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(report["diagnostic_flags"], indent=2))
    print(json.dumps(report["production_summary"], indent=2))
    for trial in report["trial_results"]:
        print(trial["target"], {key: value["region"] for key, value in trial.items()
              if isinstance(value, dict) and "region" in value})


if __name__ == "__main__":
    main()
