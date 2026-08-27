from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.hand.gesture_classifier import classify_pointing
from src.safety.path_guard import PROJECT_ROOT


@dataclass
class HandProbeResult:
    detected: bool = False
    normalized_landmarks: np.ndarray | None = None
    pixel_landmarks: np.ndarray | None = None
    world_landmarks_m: np.ndarray | None = None
    handedness: str | None = None
    handedness_score: float | None = None
    is_pointing: bool = False
    gesture_confidence: float = 0.0


class MediaPipeHandProbe:
    """Evaluation-only access to every output already produced by the task model."""

    def __init__(self,max_hands=1,detection_confidence=.55,tracking_confidence=.55,
                 landmark_ema_alpha=.55):
        import mediapipe as mp
        model_path=PROJECT_ROOT/"models"/"mediapipe"/"hand_landmarker.task"
        base=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
        options=mp.tasks.vision.HandLandmarkerOptions(
            base_options=base,running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        self._mp=mp
        self._detector=mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms=0
        self._alpha=float(landmark_ema_alpha)
        self._normalized_ema=None
        self._world_ema=None

    def _smooth(self,current,previous):
        return current if previous is None else self._alpha*current+(1-self._alpha)*previous

    def process(self,frame_bgr):
        height,width=frame_bgr.shape[:2]
        rgb=np.ascontiguousarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
        image=self._mp.Image(image_format=self._mp.ImageFormat.SRGB,data=rgb)
        self._timestamp_ms+=1
        result=self._detector.detect_for_video(image,self._timestamp_ms)
        if not result.hand_landmarks:
            self._normalized_ema=self._world_ema=None
            return HandProbeResult()
        normalized=np.asarray([[point.x,point.y,point.z]
                               for point in result.hand_landmarks[0]],dtype=np.float32)
        self._normalized_ema=self._smooth(normalized,self._normalized_ema)
        normalized=self._normalized_ema.copy()
        pixels=normalized.copy()
        pixels[:,0]*=width; pixels[:,1]*=height; pixels[:,2]*=width
        world=None
        if result.hand_world_landmarks:
            raw_world=np.asarray([[point.x,point.y,point.z]
                                  for point in result.hand_world_landmarks[0]],dtype=np.float32)
            self._world_ema=self._smooth(raw_world,self._world_ema)
            world=self._world_ema.copy()
        handedness=score=None
        if result.handedness and result.handedness[0]:
            category=result.handedness[0][0]
            handedness=category.category_name
            score=float(category.score)
        is_pointing,gesture_confidence=classify_pointing(pixels)
        return HandProbeResult(True,normalized,pixels,world,handedness,score,
                               is_pointing,float(gesture_confidence))

    def close(self):
        self._detector.close()
