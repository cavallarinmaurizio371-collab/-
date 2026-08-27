from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.direction_candidates.hand_probe import MediaPipeHandProbe
from src.camera.camera import Camera
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_estimator import DepthEstimator
from src.experimental_3d_pointing.generic_detector import (
    GenericObjectDetector, coco_supported_classes, resolve_object_class_configuration,
    robust_bbox_depth,
)
from src.experimental_3d_pointing.intrinsics import (
    backproject_distorted_pixel, load_phase2b_intrinsics, project_camera_points,
)
from src.experimental_3d_pointing.object_selection import (
    ObjectSelectionHysteresis, select_object_by_ray,
)
from src.experimental_3d_pointing.unified_core import UnifiedFingerRayCore, UnifiedHandState
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path
from src.visualization.renderer import CONNECTIONS


REGION_GRID = (
    ("LEFT_UP", "UP", "RIGHT_UP"),
    ("LEFT", "CENTER", "RIGHT"),
    ("LEFT_DOWN", "DOWN", "RIGHT_DOWN"),
)


def _text(image, value, position, color=(255, 255, 255), scale=.48, thickness=1):
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _display_point(point, width, mirror):
    x, y = int(round(point[0])), int(round(point[1]))
    return (width - 1 - x if mirror else x, y)


def _display_bbox(bbox, width, mirror):
    x1, y1, x2, y2 = bbox
    return (width - 1 - x2, y1, width - 1 - x1, y2) if mirror else bbox


def _draw_hand(display, probe, mirror):
    if not probe.detected: return
    width = display.shape[1]
    points = [_display_point(point, width, mirror) for point in probe.pixel_landmarks[:, :2]]
    for start, end in CONNECTIONS:
        cv2.line(display, points[start], points[end], (70, 230, 95), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        highlighted = index in (5, 6, 7, 8)
        cv2.circle(display, point, 6 if highlighted else 3,
                   (0, 230, 255) if highlighted else (60, 255, 100), -1)


def _draw_direction(display, probe, ray, intrinsics, mirror):
    if not probe.detected or ray.direction_camera is None: return
    width = display.shape[1]
    if ray.origin_camera is not None:
        end = ray.origin_camera + .25 * ray.direction_camera
        if ray.origin_camera[2] > .02 and end[2] > .02:
            try:
                pixels = project_camera_points([ray.origin_camera, end], intrinsics)
                start = _display_point(pixels[0], width, mirror)
                finish = _display_point(pixels[1], width, mirror)
                cv2.arrowedLine(display, start, finish, (20, 30, 255), 4,
                                cv2.LINE_AA, tipLength=.08)
                return
            except ValueError:
                pass
    # Direction-only fallback is visualization, not business geometry.
    tip = np.asarray(probe.pixel_landmarks[8, :2], dtype=float)
    pip = np.asarray(probe.pixel_landmarks[6, :2], dtype=float)
    visible = tip - pip; norm = float(np.linalg.norm(visible))
    if norm > 1e-6:
        finish = tip + visible / norm * 220.0
        cv2.arrowedLine(display, _display_point(tip, width, mirror),
                        _display_point(finish, width, mirror), (20, 30, 255), 4,
                        cv2.LINE_AA, tipLength=.08)


def _draw_objects(display, objects, selected_id, mirror):
    width = display.shape[1]
    for detected in objects:
        selected = detected.id == selected_id
        x1, y1, x2, y2 = _display_bbox(detected.bbox, width, mirror)
        color = (0, 255, 255) if selected else (255, 155, 35)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 4 if selected else 2)
        distance = "N/A" if detected.distance_m is None else f"~{detected.distance_m:.2f}m"
        _text(display, f"{detected.class_name} #{detected.id} {detected.confidence:.2f} {distance}",
              (x1, max(112, y1 - 7)), color, .42, 1)


def _draw_region_panel(display, region):
    height, width = display.shape[:2]; panel_width, panel_height = 420, 285
    left, top = width - panel_width - 16, 118
    cell_width, cell_height = panel_width // 3, panel_height // 3
    cv2.rectangle(display, (left, top), (left + panel_width, top + panel_height),
                  (25, 25, 25), -1)
    for row, names in enumerate(REGION_GRID):
        for column, name in enumerate(names):
            x0 = left + column * cell_width; y0 = top + row * cell_height
            x1 = left + (column + 1) * cell_width; y1 = top + (row + 1) * cell_height
            cv2.rectangle(display, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2),
                          (20, 145, 240) if name == region else (40, 40, 40), -1)
            cv2.rectangle(display, (x0, y0), (x1, y1), (205, 205, 205), 1)
            size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0]
            _text(display, name, (x0 + (cell_width - size[0]) // 2,
                                  y0 + (cell_height + size[1]) // 2),
                  (255, 255, 255), .45)
    _text(display, "REGION = USER PHYSICAL DIRECTION",
          (left + 35, top + panel_height + 25), (0, 255, 255), .43)


def _object_depth(objects, depth_map, intrinsics, depth_calibration, cache, config):
    alpha = float(config["ema_alpha"])
    for detected in objects:
        sampled = robust_bbox_depth(depth_map, detected.bbox,
            config["bbox_inner_ratio"], config["lower_percentile"],
            config["upper_percentile"], config["outlier_mad_scale"])
        raw = sampled.median_m if sampled.valid else None
        corrected = depth_calibration.correct(raw) if raw is not None else None
        detected.raw_depth_m = raw
        if corrected is None or not np.isfinite(corrected) or corrected <= 0:
            continue
        key = (detected.class_id, detected.id)
        old = cache.get(key, float(corrected)); filtered = alpha * float(corrected) + (1 - alpha) * old
        cache[key] = filtered; detected.depth_m = filtered
        try:
            detected.center_camera = backproject_distorted_pixel(
                *detected.center_2d, filtered, intrinsics)
            detected.distance_m = float(np.linalg.norm(detected.center_camera))
            detected.depth_valid = True
        except ValueError:
            detected.depth_valid = False


def _fmt_vector(value):
    if value is None: return "N/A"
    return "(" + ", ".join(f"{float(item):+.3f}" for item in value) + ")"


def main():
    parser = argparse.ArgumentParser(description="Unified Camera-space 3D pointing delivery demo")
    parser.add_argument("--config", default="configs/last_app.yaml")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--depth", choices=("auto", "metric", "approximate"), default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    project = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    config_path = assert_safe_path(PROJECT_ROOT / args.config)
    config = load_yaml(config_path)
    camera_index = project["camera"]["index"] if args.camera is None else args.camera
    width = int(config["intrinsics"]["required_width"]); height = int(config["intrinsics"]["required_height"])
    intrinsics = load_phase2b_intrinsics(PROJECT_ROOT, {"intrinsics": {
        "calibrated_file": config["intrinsics"]["calibrated_file"]}}, width, height)
    camera = Camera(camera_index, width, height, project["camera"]["fps"])
    hand = MediaPipeHandProbe(**project["hand"])
    object_config = config["object_detection"]
    supported_classes = coco_supported_classes()
    class_config = resolve_object_class_configuration(
        supported_classes, object_config["enabled"], object_config["target_classes"],
        object_config["excluded_classes"], object_config["use_all_supported_classes"])
    detector = GenericObjectDetector(**{
        key: value for key, value in object_config.items() if key != "enabled"
    }) if class_config.enabled else None
    depth_config = project["depth"]
    depth = DepthEstimator(depth_config["model_id"], args.depth or depth_config["backend"],
                           depth_config["fallback_depth_m"])
    depth_calibration = DepthCalibration.load(PROJECT_ROOT / depth_config["calibration_file"])
    ray_core = UnifiedFingerRayCore(config)
    selection_config = config["selection"]
    selection_temporal = ObjectSelectionHysteresis(
        selection_config["switch_confirm_frames"], selection_config["release_frames"])
    mirror = bool(config["display"]["mirror"]); debug = bool(config["display"]["debug"])
    depth_cache = {}; previous = time.perf_counter(); fps = 0.0; frame_count = 0
    print("Detector Backend: SSDLite320 MobileNetV3 Large (COCO)")
    print(f"Supported Classes ({len(supported_classes)}):", ", ".join(supported_classes))
    if not class_config.enabled:
        print("Enabled Target Classes: NONE (object detection/selection disabled)")
    elif class_config.all_classes:
        print(f"Enabled Target Classes: ALL ({len(class_config.included_classes)} classes)")
    else:
        print("Enabled Target Classes:", ", ".join(class_config.included_classes))
    print("Excluded Classes:", ", ".join(class_config.excluded_classes) or "NONE")
    for unknown in class_config.unknown_classes:
        print(f"WARNING: configured target class is not supported by detector: {unknown}")
    print("Camera intrinsics:", intrinsics.mode)
    try:
        camera.open(); ok, frame = camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        if frame.shape[:2] != (height, width):
            raise RuntimeError(f"CALIBRATION_RESOLUTION_MISMATCH: got {frame.shape[1]}x{frame.shape[0]}")
        while ok:
            now = time.perf_counter(); interval = max(now - previous, 1e-6); previous = now
            fps = .9 * fps + .1 / interval if fps else 1.0 / interval
            probe = UnifiedHandState.from_probe(hand.process(frame))
            objects = detector.process(frame) if detector is not None else []
            depth_map = depth.process(frame)
            corrector = depth_calibration.correct if depth_calibration.calibrated else None
            ray = ray_core.process(probe, depth_map, intrinsics, corrector)
            _object_depth(objects, depth_map, intrinsics, depth_calibration, depth_cache,
                          config["object_depth"])
            proposed = metric = None
            if ray.gesture_active and ray.status == "POINTING_VALID" \
                    and ray.origin_camera is not None and ray.direction_camera is not None:
                proposed, metric, _ = select_object_by_ray(
                    objects, ray.origin_camera, ray.direction_camera, intrinsics,
                    **{key: value for key, value in selection_config.items()
                       if key not in ("switch_confirm_frames", "release_frames")})
            selected_id = selection_temporal.update(proposed.id if proposed is not None else None)
            selected = next((item for item in objects if item.id == selected_id), None)
            status = "TARGET_SELECTED" if selected is not None else ray.status

            display = cv2.flip(frame, 1) if mirror else frame.copy()
            cv2.rectangle(display, (0, 0), (width, 102), (20, 20, 20), -1)
            _draw_hand(display, probe, mirror); _draw_direction(display, probe, ray, intrinsics, mirror)
            _draw_objects(display, objects, selected_id, mirror); _draw_region_panel(display, ray.stable_region)
            gesture_text = "POINTING" if ray.gesture_active else ("HAND" if probe.detected else "NO HAND")
            object_text = "NONE" if selected is None else f"{selected.class_name} #{selected.id}"
            object_distance = "N/A" if selected is None or selected.distance_m is None else f"~{selected.distance_m:.2f} m"
            finger_distance = "N/A" if ray.finger_distance_m is None else f"~{ray.finger_distance_m:.2f} m"
            region_text = ray.stable_region or "NO_STABLE_POINTING"
            quality_text = "GOOD" if ray.orientation_quality == "GOOD" else "3D DIRECTION UNRELIABLE"
            _text(display, f"Gesture: {gesture_text} | Status: {status} | Direction Quality: {quality_text}",
                  (14, 30), (0, 255, 255), .58, 2)
            _text(display, f"Pointing Region: {region_text} | Selected Object: {object_text} | Object Distance: {object_distance}",
                  (14, 58), (230, 230, 230), .52, 1)
            _text(display, f"Finger Distance: {finger_distance} | FPS: {fps:.1f} | {intrinsics.mode} | M mirror D debug R reset Q quit",
                  (14, 86), (210, 210, 210), .43, 1)
            if debug:
                panel_top = 430
                cv2.rectangle(display, (12, panel_top), (width - 12, height - 8), (20, 20, 20), -1)
                _text(display, f"Orientation={ray.orientation_name} quality={ray.orientation_quality} PnP RMSE={ray.pnp_rmse_px}",
                      (22, panel_top + 25), (220, 220, 220), .42)
                _text(display, f"B={_fmt_vector(ray.candidate_b_camera)} C={_fmt_vector(ray.candidate_c_camera)} selected={ray.direction_source}",
                      (22, panel_top + 50), (220, 220, 220), .42)
                _text(display, f"Origin XYZ={_fmt_vector(ray.origin_camera)} Direction XYZ={_fmt_vector(ray.direction_camera)}",
                      (22, panel_top + 75), (220, 220, 220), .42)
                hit_text = "N/A" if ray.coarse_hit is None or not ray.coarse_hit.valid else \
                    f"yaw={ray.coarse_hit.yaw_deg:+.2f} pitch={ray.coarse_hit.pitch_deg:+.2f} raw={ray.coarse_hit.region}"
                _text(display, f"Normalized hit: {hit_text} | Anchor={ray.anchor_status} Z={ray.anchor_depth_m}",
                      (22, panel_top + 100), (220, 220, 220), .42)
                _text(display, f"Object selected metric={metric} | Depth={depth.mode} | scene depth only supplies global XYZ",
                      (22, panel_top + 125), (220, 220, 220), .42)
                for row, detected in enumerate(objects[:4]):
                    _text(display, f"Object {detected.id} {detected.class_name}: XYZ={_fmt_vector(detected.center_camera)} depth_valid={detected.depth_valid}",
                          (22, panel_top + 150 + row * 22), (170, 210, 255), .38)

            frame_count += 1
            if not args.headless:
                cv2.imshow("Unified 3D Pointing - Final Delivery", display)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q"), ord("Q")): break
                if key in (ord("m"), ord("M")): mirror = not mirror
                elif key in (ord("d"), ord("D")): debug = not debug
                elif key in (ord("r"), ord("R")):
                    ray_core.reset(); selection_temporal.reset(); depth_cache.clear()
            if args.max_frames and frame_count >= args.max_frames: break
            ok, frame = camera.read()
    finally:
        hand.close(); camera.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
