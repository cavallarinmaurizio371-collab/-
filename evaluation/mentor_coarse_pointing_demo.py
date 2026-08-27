from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.direction_candidates import fit_finger_axis_with_quality
from evaluation.direction_candidates.hand_probe import MediaPipeHandProbe
from src.camera.camera import Camera
from src.depth.depth_estimator import DepthEstimator
from src.depth.depth_sampler import sample_point
from src.experimental_3d_pointing.coarse import (
    CoarseTemporalStabilizer,
    normalized_camera_plane_hit,
    select_candidate_no_gt,
    trial_coarse_median,
)
from src.experimental_3d_pointing.intrinsics import load_phase2b_intrinsics
from src.experimental_3d_pointing.orientation import (
    OrientationSelector,
    direction_to_camera,
    solve_orientation_hypotheses,
)
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path, safe_mkdir
from src.visualization.renderer import CONNECTIONS


REGIONS = (
    ("LEFT_UP", "UP", "RIGHT_UP"),
    ("LEFT", "CENTER", "RIGHT"),
    ("LEFT_DOWN", "DOWN", "RIGHT_DOWN"),
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _text(image, value, position, color=(255, 255, 255), scale=.5, thickness=1):
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _display_point(point, width, mirror):
    x, y = int(round(point[0])), int(round(point[1]))
    return (width - 1 - x if mirror else x, y)


def _draw_hand(image, probe, mirror):
    if not probe.detected:
        return
    points = np.asarray(probe.pixel_landmarks[:, :2], dtype=float)
    width = image.shape[1]
    shown = [_display_point(point, width, mirror) for point in points]
    for start, end in CONNECTIONS:
        cv2.line(image, shown[start], shown[end], (65, 225, 95), 2, cv2.LINE_AA)
    for index, point in enumerate(shown):
        highlighted = index in (5, 6, 7, 8)
        cv2.circle(image, point, 6 if highlighted else 3,
                   (0, 230, 255) if highlighted else (60, 255, 100), -1)


def _draw_direction(image, probe, direction, intrinsics, mirror):
    if not probe.detected or direction is None:
        return
    tip = np.asarray(probe.pixel_landmarks[8, :2], dtype=float)
    normalized_tip = cv2.undistortPoints(
        np.asarray(tip, dtype=np.float64).reshape(1, 1, 2),
        np.asarray([[intrinsics.fx, 0, intrinsics.cx],
                    [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=float),
        np.asarray(intrinsics.distortion, dtype=float),
    ).reshape(2)
    dx, dy, dz = np.asarray(direction, dtype=float)
    tangent = np.asarray([dx - normalized_tip[0] * dz,
                          dy - normalized_tip[1] * dz], dtype=float)
    tangent_px = tangent * np.asarray([intrinsics.fx, intrinsics.fy], dtype=float)
    norm = float(np.linalg.norm(tangent_px))
    if norm <= 1e-6:
        return
    end = tip + tangent_px / norm * 230.0
    start_display = _display_point(tip, image.shape[1], mirror)
    end_display = _display_point(end, image.shape[1], mirror)
    cv2.arrowedLine(image, start_display, end_display, (20, 30, 255), 4,
                    cv2.LINE_AA, tipLength=.08)


def _draw_panel(image, selected_region):
    height, width = image.shape[:2]
    panel_width, panel_height = 480, 330
    left, top = width - panel_width - 18, 105
    cell_width, cell_height = panel_width // 3, panel_height // 3
    cv2.rectangle(image, (left, top), (left + panel_width, top + panel_height),
                  (24, 24, 24), -1)
    for row, names in enumerate(REGIONS):
        for column, name in enumerate(names):
            x0 = left + column * cell_width; y0 = top + row * cell_height
            x1 = left + (column + 1) * cell_width; y1 = top + (row + 1) * cell_height
            active = name == selected_region
            fill = (20, 145, 240) if active else (38, 38, 38)
            cv2.rectangle(image, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), fill, -1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (210, 210, 210), 1)
            size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, .52, 1)[0]
            _text(image, name, (x0 + (cell_width - size[0]) // 2,
                                y0 + (cell_height + size[1]) // 2),
                  (255, 255, 255), .52, 1)
    _text(image, "REGION LABELS = USER PHYSICAL DIRECTIONS",
          (left + 20, top + panel_height + 28), (0, 255, 255), .43, 1)


class ValidationRecorder:
    def __init__(self, root, targets):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_id = f"session_{timestamp}"
        self.directory = safe_mkdir(PROJECT_ROOT / root / self.session_id)
        self.target_names = list(targets); self.trials = []

    def save_trial(self, target, rows, threshold_x, threshold_y, minimum_frames):
        valid_hits = [row["selected_hit_object"] for row in rows
                      if row.get("selected_hit_object") is not None]
        result = trial_coarse_median(valid_hits, threshold_x, threshold_y)
        trial = {
            "target": target, "frame_count": len(rows),
            "valid_frame_count": len(valid_hits),
            "valid": bool(result.valid and len(valid_hits) >= int(minimum_frames)),
            "prediction": result.region if result.valid else None,
            "yaw_median_deg": result.yaw_deg, "pitch_median_deg": result.pitch_deg,
        }
        self.trials.append(trial)
        frame_path = assert_safe_path(self.directory / f"{target}_frames.jsonl")
        with frame_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                clean = {key: value for key, value in row.items()
                         if key != "selected_hit_object"}
                handle.write(json.dumps(_jsonable(clean), ensure_ascii=False) + "\n")
        return trial

    def close(self):
        valid = [trial for trial in self.trials if trial["valid"]]
        correct = [trial for trial in valid if trial["prediction"] == trial["target"]]
        summary = {
            "session_id": self.session_id,
            "protocol": "MINIMAL_FIVE_TRIAL_DIRECTION_CHECK",
            "trials": self.trials,
            "trial_valid_rate": len(valid) / len(self.trials) if self.trials else 0.0,
            "accuracy": len(correct) / len(valid) if valid else None,
            "note": "GT was used only for evaluation; never for orientation/candidate selection.",
        }
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary


def _orientation_debug(estimates):
    result = {}
    for name, estimate in estimates.items():
        result[name] = {
            "valid": estimate.valid, "status": estimate.status,
            "rmse_px": estimate.reprojection_rmse_px,
            "inlier_count": estimate.inlier_count,
            "inlier_rate": estimate.inlier_rate,
            "rvec": estimate.rvec.tolist() if estimate.rvec is not None else None,
            "tvec": estimate.translation.tolist() if estimate.translation is not None else None,
            "rotation_magnitude_deg": estimate.rotation_magnitude_deg,
            "projection_agreement": estimate.projection_agreement,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Camera-centered coarse 3D pointing mentor demo")
    parser.add_argument("--validate-five", action="store_true",
                        help="Record CENTER/LEFT/RIGHT/UP/DOWN once each")
    parser.add_argument("--depth", choices=("metric", "auto", "approximate"), default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    project = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    config = load_yaml(PROJECT_ROOT / "configs/mentor_coarse_pointing.yaml")
    width = int(config["intrinsics"]["required_width"])
    height = int(config["intrinsics"]["required_height"])
    intrinsics_config = {"intrinsics": {
        "calibrated_file": config["intrinsics"]["calibrated_file"]
    }}
    intrinsics = load_phase2b_intrinsics(PROJECT_ROOT, intrinsics_config, width, height)
    camera = Camera(project["camera"]["index"], width, height, project["camera"]["fps"])
    hand = MediaPipeHandProbe(**project["hand"])
    depth_backend = args.depth or config["display"]["depth_backend"]
    depth_cfg = project["depth"]
    depth = DepthEstimator(depth_cfg["model_id"], depth_backend,
                           depth_cfg["fallback_depth_m"])
    selector = OrientationSelector(config["orientation"]["selection"])
    temporal_cfg = config["temporal"]; region_cfg = config["coarse_region"]
    temporal = CoarseTemporalStabilizer(
        temporal_cfg["window_size"], temporal_cfg["min_stable_frames"],
        temporal_cfg["hold_frames"], temporal_cfg["max_jitter_deg"],
        region_cfg["center_half_angle_x_deg"], region_cfg["center_half_angle_y_deg"],
    )
    mirror = bool(config["display"]["mirror"]); debug = bool(config["display"]["debug"])
    recorder = ValidationRecorder(config["validation"]["output_root"],
                                  config["validation"]["targets"]) if args.validate_five else None
    target_index = 0; validation_state = "IDLE"; deadline = 0.0; trial_rows = []
    approximate_distance = None; frame_index = 0; previous_time = time.perf_counter(); fps = 0.0
    selected_direction = None; final_summary = None
    try:
        camera.open()
        ok, frame = camera.read()
        if not ok or frame.shape[:2] != (height, width):
            raise RuntimeError("Camera must provide the calibrated 1280x720 resolution")
        while ok:
            now = time.perf_counter(); elapsed = max(now - previous_time, 1e-6)
            previous_time = now; fps = .9 * fps + .1 / elapsed if fps else 1.0 / elapsed
            probe = hand.process(frame)
            estimates = {}; orientation_name = "ORIENTATION_UNRELIABLE"
            orientation_quality = "UNRELIABLE"; source = "INVALID"
            hit_b = hit_c = selected_hit = None; axis = None; selected_direction = None
            if probe.detected and probe.world_landmarks_m is not None:
                world = np.asarray(probe.world_landmarks_m, dtype=float)
                b_native = world[8] - world[6]
                axis = fit_finger_axis_with_quality(world)
                c_native = axis.direction if axis.valid else None
                estimates = solve_orientation_hypotheses(
                    world, probe.pixel_landmarks, intrinsics, config["orientation"],
                    {"B": b_native, "C": c_native},
                )
                orientation_candidate = "C" if axis.valid else "B"
                orientation_name, orientation, orientation_quality = selector.select(
                    estimates, orientation_candidate
                )
                if orientation is not None:
                    direction_b = direction_to_camera(b_native, orientation)
                    direction_c = direction_to_camera(c_native, orientation)
                    hit_b = normalized_camera_plane_hit(
                        probe.pixel_landmarks[8, :2], direction_b, intrinsics,
                        region_cfg["center_half_angle_x_deg"],
                        region_cfg["center_half_angle_y_deg"],
                        region_cfg["intersection_epsilon"],
                    )
                    hit_c = normalized_camera_plane_hit(
                        probe.pixel_landmarks[8, :2], direction_c, intrinsics,
                        region_cfg["center_half_angle_x_deg"],
                        region_cfg["center_half_angle_y_deg"],
                        region_cfg["intersection_epsilon"],
                    )
                    axis_valid = bool(axis.valid and
                        axis.linearity >= float(config["candidate"]["c_min_linearity"]) and
                        axis.residual_m <= float(config["candidate"]["c_max_residual_m"]))
                    c_quality = "GOOD" if axis_valid else "HAND_AXIS_UNSTABLE"
                    b_quality = "GOOD" if direction_b is not None else "INVALID"
                    source, selected_hit = select_candidate_no_gt(
                        hit_c, hit_b, c_quality, b_quality, axis_valid,
                        tuple(config["candidate"]["accepted_quality"]),
                    )
                    selected_direction = direction_c if source == "C" else (
                        direction_b if source == "B_FALLBACK" else None
                    )
            temporal_status, stable_region, stable_angles, jitter = temporal.update(
                selected_hit, probe.is_pointing and orientation_quality == "GOOD"
            )

            update_every = int(config["display"]["distance_update_every_frames"])
            if probe.detected and frame_index % max(update_every, 1) == 0:
                depth_map = depth.process(frame)
                values = [sample_point(depth_map, probe.pixel_landmarks[i, :2], 7)
                          for i in (0, 5, 9, 13, 17)]
                finite = [float(v) for v in values if v is not None and np.isfinite(v) and v > 0]
                approximate_distance = float(np.median(finite)) if finite else None

            if recorder is not None and target_index < len(recorder.target_names):
                target = recorder.target_names[target_index]
                if validation_state == "PREPARING" and now >= deadline:
                    validation_state = "COLLECTING"
                    deadline = now + float(config["validation"]["collect_seconds"])
                    trial_rows = []
                if validation_state == "COLLECTING":
                    trial_rows.append({
                        "timestamp": datetime.now().isoformat(), "target": target,
                        "pointing": probe.is_pointing,
                        "image_landmarks_px": probe.pixel_landmarks.tolist() if probe.detected else None,
                        "world_landmarks_m": probe.world_landmarks_m.tolist()
                            if probe.world_landmarks_m is not None else None,
                        "tip_pixel_raw": probe.pixel_landmarks[8, :2].tolist()
                            if probe.detected else None,
                        "orientation_selected": orientation_name,
                        "orientation_quality": orientation_quality,
                        "orientation_hypotheses": _orientation_debug(estimates),
                        "candidate_selected": source,
                        "B_hit": hit_b.__dict__ if hit_b is not None else None,
                        "C_hit": hit_c.__dict__ if hit_c is not None else None,
                        "selected_hit": selected_hit.__dict__ if selected_hit is not None else None,
                        "selected_hit_object": selected_hit,
                    })
                    if now >= deadline:
                        trial = recorder.save_trial(
                            target, trial_rows,
                            region_cfg["center_half_angle_x_deg"],
                            region_cfg["center_half_angle_y_deg"],
                            config["validation"]["minimum_valid_frames"],
                        )
                        print("Saved validation trial:", trial)
                        target_index += 1; validation_state = "IDLE"; trial_rows = []
                        temporal.reset(); selector.reset()

            display = cv2.flip(frame, 1) if mirror else frame.copy()
            cv2.rectangle(display, (0, 0), (width, 94), (20, 20, 20), -1)
            _draw_hand(display, probe, mirror)
            _draw_direction(display, probe, selected_direction, intrinsics, mirror)
            cv2.drawMarker(display, (width // 2, height // 2), (0, 255, 255),
                           cv2.MARKER_CROSS, 28, 2)
            _draw_panel(display, stable_region)
            headline = stable_region if stable_region else "NO_STABLE_POINTING"
            _text(display, f"POINTING: {headline}", (18, 39),
                  (0, 255, 255) if stable_region else (0, 170, 255), 1.0, 2)
            distance_text = "N/A" if approximate_distance is None else f"{approximate_distance:.2f} m"
            _text(display, f"Source: {source} | Orientation: {orientation_quality} ({orientation_name}) | FPS {fps:.1f}",
                  (18, 68), (230, 230, 230), .48)
            _text(display, f"Approx Distance: {distance_text} | Mirror {'ON' if mirror else 'OFF'} | D debug | M mirror | Q quit",
                  (18, 90), (210, 210, 210), .43)
            if recorder is not None:
                if target_index < len(recorder.target_names):
                    target = recorder.target_names[target_index]
                    remain = max(0.0, deadline - now) if validation_state != "IDLE" else 0.0
                    _text(display, f"5-TRIAL CHECK {target_index + 1}/5: physically point {target} | {validation_state} {remain:.1f}s | SPACE start",
                          (18, height - 22), (0, 255, 255), .5, 1)
                else:
                    _text(display, "5-TRIAL CHECK COMPLETE - press Q to save/exit",
                          (18, height - 22), (0, 255, 0), .55, 2)
            if debug:
                top = 125
                cv2.rectangle(display, (12, top - 25), (610, top + 190), (20, 20, 20), -1)
                for name in ("R_PALM_STABLE", "R21_RANSAC", "R21_CURRENT", "RAW_WORLD_DIAGNOSTIC"):
                    estimate = estimates.get(name)
                    if estimate is None:
                        line = f"{name}: N/A"
                    else:
                        agreement = estimate.projection_agreement.get("C", {})
                        line = (f"{name}: {estimate.status} rmse={estimate.reprojection_rmse_px} "
                                f"inliers={estimate.inlier_count}/{estimate.inlier_rate} "
                                f"agree={agreement}")
                    _text(display, line, (22, top), (220, 220, 220), .38)
                    top += 34
                if selected_hit is not None:
                    _text(display, f"Normalized hit yaw/pitch={selected_hit.yaw_deg:.2f}/{selected_hit.pitch_deg:.2f} "
                          f"raw={selected_hit.region} temporal={temporal_status} jitter={jitter}",
                          (22, top), (0, 220, 255), .40)

            frame_index += 1
            if not args.headless:
                cv2.imshow("Mentor Camera-Centered Coarse Pointing", display)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("d"), ord("D")):
                    debug = not debug
                elif key in (ord("m"), ord("M")):
                    mirror = not mirror
                elif key == 32 and recorder is not None and validation_state == "IDLE" \
                        and target_index < len(recorder.target_names):
                    temporal.reset(); selector.reset()
                    validation_state = "PREPARING"
                    deadline = now + float(config["validation"]["prepare_seconds"])
            if args.max_frames and frame_index >= args.max_frames:
                break
            ok, frame = camera.read()
    finally:
        if recorder is not None:
            final_summary = recorder.close()
            print("Validation output:", recorder.directory)
            print("Validation summary:", final_summary)
        hand.close(); camera.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
