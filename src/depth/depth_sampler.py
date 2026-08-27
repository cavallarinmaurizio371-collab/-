from __future__ import annotations

import numpy as np


def _median_valid(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


def sample_point(depth_map: np.ndarray, point, patch_size=7) -> float | None:
    if depth_map is None or point is None:
        return None
    x, y = [int(round(v)) for v in point]
    radius = max(1, int(patch_size) // 2)
    h, w = depth_map.shape[:2]
    return _median_valid(depth_map[max(0, y-radius):min(h, y+radius+1),
                                   max(0, x-radius):min(w, x+radius+1)])


def sample_bbox(depth_map: np.ndarray, bbox, inner_ratio=0.5) -> float | None:
    if depth_map is None:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    hw, hh = (x2-x1)*inner_ratio/2, (y2-y1)*inner_ratio/2
    h, w = depth_map.shape[:2]
    xa, xb = max(0, int(cx-hw)), min(w, int(cx+hw)+1)
    ya, yb = max(0, int(cy-hh)), min(h, int(cy+hh)+1)
    return _median_valid(depth_map[ya:yb, xa:xb])

