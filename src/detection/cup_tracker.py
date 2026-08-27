from __future__ import annotations

import numpy as np

from src.types import CupDetection


class CupTracker:
    def __init__(self, max_distance_px=150, ttl_frames=20):
        self.max_distance = float(max_distance_px)
        self.ttl = int(ttl_frames)
        self.tracks: dict[int, tuple[np.ndarray, int]] = {}
        self.next_id = 1

    def update(self, detections: list[CupDetection]) -> list[CupDetection]:
        unmatched = set(self.tracks)
        for det in sorted(detections, key=lambda d: d.center_2d[0]):
            center = np.asarray(det.center_2d, dtype=float)
            candidates = [(float(np.linalg.norm(center - c)), track_id)
                          for track_id, (c, age) in self.tracks.items() if track_id in unmatched]
            distance, match = min(candidates, default=(float("inf"), None))
            if match is not None and distance <= self.max_distance:
                det.id = match
                det.track_age = self.tracks[match][1] + 1
                unmatched.remove(match)
            else:
                det.id = self.next_id
                self.next_id += 1
            self.tracks[det.id] = (center, det.track_age)
        for track_id in list(unmatched):
            center, age = self.tracks[track_id]
            if age >= self.ttl:
                del self.tracks[track_id]
            else:
                self.tracks[track_id] = (center, age + 1)
        # User-facing labels are left-to-right while internal association stays stable.
        for label, det in enumerate(sorted(detections, key=lambda d: d.center_2d[0]), 1):
            det.id = label
        return detections

