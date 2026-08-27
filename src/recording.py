from __future__ import annotations

import csv
import json
from datetime import datetime

import cv2

from src.safety.path_guard import PROJECT_ROOT, assert_safe_path, safe_mkdir, safe_open


class SessionRecorder:
    def __init__(self, enabled, frame_size, fps=20):
        self.enabled, self.frames = enabled, 0
        self.writer = self.events = self.csv = None
        if not enabled:
            return
        self.directory = safe_mkdir(PROJECT_ROOT / "outputs" / datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        video_path = assert_safe_path(self.directory / "demo.mp4")
        self.writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
        self.events = safe_open(self.directory / "events.csv", "w", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(self.events, fieldnames=["timestamp","is_pointing","hand_x","hand_y","hand_z","cup_count","selected_cup","selected_score","fps"])
        self.csv.writeheader()

    def write(self, frame, result, fps):
        if not self.enabled:
            return
        self.writer.write(frame)
        xyz = result.tip_3d if result.tip_3d is not None else (None,None,None)
        self.csv.writerow({"timestamp": datetime.now().isoformat(), "is_pointing": result.hand.is_pointing,
                           "hand_x": xyz[0], "hand_y": xyz[1], "hand_z": xyz[2],
                           "cup_count": len(result.cups), "selected_cup": result.selected_cup,
                           "selected_score": result.selected_score, "fps": fps})
        self.frames += 1

    def close(self, average_fps=0):
        if not self.enabled:
            return
        self.writer.release(); self.events.close()
        with safe_open(self.directory / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"frames": self.frames, "average_fps": average_fps}, handle, indent=2)
