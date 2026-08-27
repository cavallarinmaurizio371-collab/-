from __future__ import annotations

import cv2
import numpy as np

from src.hand.gesture_classifier import classify_pointing
from src.safety.path_guard import PROJECT_ROOT
from src.types import HandState


class HandLandmarker:
    def __init__(self, max_hands=1, detection_confidence=0.55, tracking_confidence=0.55,
                 landmark_ema_alpha=0.55):
        try:
            import mediapipe as mp
            model_path = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing {model_path}. Run: .venv\\Scripts\\python.exe scripts\\download_models.py")
            # Pass bytes instead of a filesystem path: MediaPipe native path handling
            # can fail when the required project root contains non-ASCII characters.
            base = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence)
            self._mp = mp
            self._hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("MediaPipe Hand Landmarker is unavailable; run scripts/setup.ps1") from exc
        self.alpha = float(landmark_ema_alpha)
        self._smoothed = None
        self._timestamp_ms = 0

    def process(self, frame_bgr) -> HandState:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 1
        result = self._hands.detect_for_video(image, self._timestamp_ms)
        if not result.hand_landmarks:
            self._smoothed = None
            return HandState()
        lm = result.hand_landmarks[0]
        points = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
        self._smoothed = points if self._smoothed is None else self.alpha * points + (1 - self.alpha) * self._smoothed
        points = self._smoothed.copy()
        is_pointing, confidence = classify_pointing(points)
        direction = points[8, :2] - points[6, :2]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)
        return HandState(True, is_pointing, confidence, points, points[5, :2], points[6, :2],
                         points[7, :2], points[8, :2], direction)

    def close(self):
        self._hands.close()
