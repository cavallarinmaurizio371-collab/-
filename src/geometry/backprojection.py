from __future__ import annotations

import math
import numpy as np


def pixel_depth_to_camera_xyz(u, v, depth, intrinsics) -> np.ndarray:
    if depth is None or not math.isfinite(float(depth)) or depth <= 0:
        raise ValueError("Depth must be finite and positive")
    z = float(depth)
    return np.array([(float(u)-intrinsics.cx)*z/intrinsics.fx,
                     (float(v)-intrinsics.cy)*z/intrinsics.fy, z], dtype=np.float32)


def point_metrics(xyz):
    xyz = np.asarray(xyz, dtype=float)
    return {"distance": float(np.linalg.norm(xyz)), "height": float(-xyz[1])}

