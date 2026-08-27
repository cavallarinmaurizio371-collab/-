from __future__ import annotations

import numpy as np


def make_ray(pip_xyz, tip_xyz):
    origin = np.asarray(tip_xyz, dtype=float)
    direction = origin - np.asarray(pip_xyz, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("PIP and TIP are too close to define a ray")
    return origin, direction / norm


def point_to_ray_distance(point, origin, direction):
    delta = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    t = float(np.dot(delta, direction))
    if t < 0:
        return float("inf"), t
    closest = origin + t * direction
    return float(np.linalg.norm(np.asarray(point) - closest)), t


def point_to_ray_distance_2d(point, origin, direction):
    return point_to_ray_distance(np.asarray(point, float), np.asarray(origin, float), np.asarray(direction, float))

