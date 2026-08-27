from __future__ import annotations

import numpy as np

from src.geometry.pointing_ray import point_to_ray_distance, point_to_ray_distance_2d


def select_target(cups, origin_2d, direction_2d, origin_3d=None, direction_3d=None,
                  weight_2d=0.68, weight_3d=0.32, max_score=1.0,
                  max_2d_distance_px=180, max_3d_distance_m=0.35):
    results = []
    for cup in cups:
        d2, t2 = point_to_ray_distance_2d(cup.center_2d, origin_2d, direction_2d)
        if t2 < 0 or not np.isfinite(d2):
            continue
        s2 = d2 / max(max_2d_distance_px, 1e-6)
        if origin_3d is not None and direction_3d is not None and cup.center_3d is not None:
            d3, t3 = point_to_ray_distance(cup.center_3d, origin_3d, direction_3d)
            if t3 < 0 or not np.isfinite(d3):
                continue
            score = weight_2d*s2 + weight_3d*(d3/max(max_3d_distance_m, 1e-6))
        else:
            score = s2
        results.append((float(score), cup.id))
    if not results:
        return None, None
    score, cup_id = min(results)
    return (cup_id, score) if score <= max_score else (None, score)


class HysteresisSelector:
    def __init__(self, switch_confirm_frames=4, release_frames=3):
        self.switch_frames, self.release_frames = int(switch_confirm_frames), int(release_frames)
        self.current = self.pending = None
        self.pending_count = self.release_count = 0

    def update(self, candidate):
        if candidate == self.current:
            self.pending, self.pending_count, self.release_count = None, 0, 0
        elif candidate is None:
            self.release_count += 1
            if self.release_count >= self.release_frames:
                self.current = None
        else:
            self.release_count = 0
            if candidate != self.pending:
                self.pending, self.pending_count = candidate, 1
            else:
                self.pending_count += 1
            if self.pending_count >= self.switch_frames:
                self.current, self.pending, self.pending_count = candidate, None, 0
        return self.current

