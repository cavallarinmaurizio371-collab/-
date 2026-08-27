from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Z0Intersection:
    valid: bool
    status: str
    point_camera: np.ndarray | None = None
    t_hit: float | None = None


def normalize_direction(direction) -> np.ndarray:
    value = np.asarray(direction, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Direction must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("Direction magnitude is too small")
    return value / norm


def intersect_ray_with_z0(origin, direction, epsilon=1e-6) -> Z0Intersection:
    """Intersect R(t)=origin+t*direction, t>=0, with the camera plane Z=0."""
    point = np.asarray(origin, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return Z0Intersection(False, "INVALID_ORIGIN")
    try:
        ray = normalize_direction(direction)
    except ValueError:
        return Z0Intersection(False, "INVALID_DIRECTION")
    if abs(float(ray[2])) < float(epsilon):
        return Z0Intersection(False, "NEAR_PARALLEL")
    t_hit = -float(point[2]) / float(ray[2])
    if t_hit < 0:
        return Z0Intersection(False, "POINTING_AWAY_FROM_CAMERA", t_hit=t_hit)
    hit = point + t_hit * ray
    hit[2] = 0.0
    return Z0Intersection(True, "VALID", hit, t_hit)


def validate_hit_range(hit_eval_mm, width_mm, height_mm) -> str:
    hit=np.asarray(hit_eval_mm,dtype=float)
    if hit.shape!=(2,) or not np.all(np.isfinite(hit)):
        return "NON_FINITE_INTERSECTION"
    if abs(float(hit[0]))>float(width_mm)/2 or abs(float(hit[1]))>float(height_mm)/2:
        return "OUT_OF_TARGET_RANGE"
    return "VALID"


def diagnostic_directions(mcp_xyz, pip_xyz, dip_xyz, tip_xyz) -> dict[str, np.ndarray | None]:
    """Extra directions for diagnosis only; baseline remains TIP-PIP."""
    points = {"mcp": mcp_xyz, "pip": pip_xyz, "dip": dip_xyz, "tip": tip_xyz}

    def between(a, b):
        if points[a] is None or points[b] is None:
            return None
        try:
            return normalize_direction(np.asarray(points[b]) - np.asarray(points[a]))
        except ValueError:
            return None

    result = {
        "tip_minus_pip": between("pip", "tip"),
        "tip_minus_dip": between("dip", "tip"),
        "dip_minus_pip": between("pip", "dip"),
        "line_fit": None,
    }
    if all(points[key] is not None for key in ("mcp", "pip", "dip", "tip")):
        matrix = np.stack([points[key] for key in ("mcp", "pip", "dip", "tip")]).astype(float)
        centered = matrix - matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        fitted = normalize_direction(vh[0])
        baseline = result["tip_minus_pip"]
        if baseline is not None and np.dot(fitted, baseline) < 0:
            fitted = -fitted
        result["line_fit"] = fitted
    return result
