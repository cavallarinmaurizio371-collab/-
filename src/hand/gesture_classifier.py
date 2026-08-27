from __future__ import annotations

import numpy as np


def _angle(a, b, c) -> float:
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def classify_pointing(points: np.ndarray) -> tuple[bool, float]:
    """Rule-based pointing classification over 21 MediaPipe landmarks."""
    if points is None or points.shape[0] < 21:
        return False, 0.0
    pip_angle = _angle(points[5], points[6], points[7])
    dip_angle = _angle(points[6], points[7], points[8])
    straightness = np.clip((min(pip_angle, dip_angle) - 125.0) / 45.0, 0.0, 1.0)
    wrist = points[0]
    index_reach = np.linalg.norm(points[8] - wrist)
    folded = []
    for pip, tip in ((10, 12), (14, 16), (18, 20)):
        folded.append(float(np.linalg.norm(points[tip] - wrist) < np.linalg.norm(points[pip] - wrist) * 1.12))
    fold_score = float(np.mean(folded))
    scale = max(np.linalg.norm(points[9] - wrist), 1e-5)
    reach_score = float(np.clip((index_reach / scale - 1.25) / 0.65, 0.0, 1.0))
    confidence = 0.55 * straightness + 0.25 * reach_score + 0.20 * fold_score
    return bool(straightness > 0.55 and confidence >= 0.58), float(confidence)

