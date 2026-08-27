from __future__ import annotations

import numpy as np


REGION_ORDER = (
    "LEFT_UP", "UP", "RIGHT_UP",
    "LEFT", "CENTER", "RIGHT",
    "LEFT_DOWN", "DOWN", "RIGHT_DOWN",
)

KEYPAD_TARGETS = {
    ord("1"): "LEFT_DOWN", ord("2"): "DOWN", ord("3"): "RIGHT_DOWN",
    ord("4"): "LEFT", ord("5"): "CENTER", ord("6"): "RIGHT",
    ord("7"): "LEFT_UP", ord("8"): "UP", ord("9"): "RIGHT_UP",
}


def camera_xyz_to_eval_xyz(point_camera) -> np.ndarray:
    """Camera(+x image-right,+y down,+z forward) -> user-facing eval axes."""
    x, y, z = np.asarray(point_camera, dtype=float)
    return np.array([-x, -y, z], dtype=float)


def camera_hit_to_eval_mm(point_camera) -> np.ndarray:
    return camera_xyz_to_eval_xyz(point_camera)[:2] * 1000.0


def configured_targets(raw_targets: dict) -> dict[str, tuple[float, float]]:
    targets = {}
    for name, value in raw_targets.items():
        if value and value.get("x_mm") is not None and value.get("y_mm") is not None:
            targets[name] = (float(value["x_mm"]), float(value["y_mm"]))
    return targets


def nearest_region(x_mm, y_mm, targets: dict[str, tuple[float, float]]) -> str | None:
    """Map by nearest physically measured target (Voronoi regions, not pixels)."""
    if not targets or not np.isfinite([x_mm, y_mm]).all():
        return None
    point = np.array([x_mm, y_mm], dtype=float)
    return min(targets, key=lambda name: float(np.linalg.norm(point - np.asarray(targets[name]))))


def mirror_display_x(x_pixel: float, frame_width: int, enabled: bool) -> float:
    """Visualization-only pixel adapter; never use for geometry or metrics."""
    return frame_width - 1 - x_pixel if enabled else x_pixel

