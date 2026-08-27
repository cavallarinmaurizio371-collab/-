from __future__ import annotations

import cv2
import numpy as np

from evaluation.coordinate_adapter import mirror_display_x
from src.visualization.renderer import CONNECTIONS

INDEX_LANDMARKS = (5, 6, 7, 8)


def display_landmarks(landmarks_2d, frame_width: int, mirror_display: bool) -> np.ndarray:
    points = np.asarray(landmarks_2d, dtype=float)[:, :2].copy()
    if mirror_display:
        points[:, 0] = frame_width - 1 - points[:, 0]
    return np.rint(points).astype(int)


def direction_display_endpoints(tip_2d, direction_2d, frame_width: int,
                                mirror_display: bool, ray_length_px=900):
    start = np.asarray(tip_2d, dtype=float)
    direction = np.asarray(direction_2d, dtype=float)
    end = start + direction * float(ray_length_px)
    start_display = np.array([
        mirror_display_x(start[0], frame_width, mirror_display), start[1]])
    end_display = np.array([
        mirror_display_x(end[0], frame_width, mirror_display), end[1]])
    return tuple(np.rint(start_display).astype(int)), tuple(np.rint(end_display).astype(int))


def direction_visual_style(intersection_status: str, intersection_valid: bool):
    if intersection_valid:
        return (30, 30, 255), "CURRENT DIRECTION"
    if intersection_status == "POINTING_AWAY_FROM_CAMERA":
        return (0, 165, 255), "AWAY"
    if intersection_status in ("INVALID_INTERSECTION", "NEAR_PARALLEL"):
        return (0, 255, 255), "NEAR PARALLEL"
    return (80, 80, 255), "CURRENT DIRECTION"


def write_png(path, image) -> bool:
    """Unicode-safe PNG output for the required Chinese project path."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return False
    encoded.tofile(path)
    return True


def draw_hand_overlay(image, result, mirror_display: bool, raw_frame_width: int,
                      ray_length_px=900) -> bool:
    """Draw the existing 21-point style and current direction; returns line visibility."""
    if not result.hand_detected or result.landmarks_2d is None:
        return False
    points = display_landmarks(result.landmarks_2d, raw_frame_width, mirror_display)
    if len(points) < 21:
        return False
    for start, end in CONNECTIONS:
        cv2.line(image, tuple(points[start]), tuple(points[end]), (100,220,100), 2)
    for index, point in enumerate(points):
        highlighted = index in INDEX_LANDMARKS
        cv2.circle(image, tuple(point), 5 if highlighted else 3,
                   (0,220,255) if highlighted else (80,255,80), -1)
    direction = result.direction_2d
    tip = result.points_2d.get("tip") if result.points_2d else None
    if not result.ray_valid or direction is None or tip is None or np.linalg.norm(direction) <= 1e-8:
        return False
    start, end = direction_display_endpoints(
        tip, direction, raw_frame_width, mirror_display, ray_length_px)
    color, label = direction_visual_style(result.intersection_status, result.intersection_valid)
    cv2.arrowedLine(image, start, end, color, 3, tipLength=0.04)
    label_pos = (max(5, min(image.shape[1]-180, start[0]+10)),
                 max(18, min(image.shape[0]-8, start[1]-12)))
    cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    return True
