from types import SimpleNamespace

import numpy as np

from src.experimental_3d_pointing.unified_core import UnifiedHandState


def test_one_probe_result_becomes_one_unified_hand_state():
    probe = SimpleNamespace(detected=True, normalized_landmarks=np.zeros((21, 3)),
        pixel_landmarks=np.ones((21, 3)), world_landmarks_m=np.full((21, 3), 2.0),
        handedness="Right", handedness_score=.9, is_pointing=True, gesture_confidence=.8)
    state = UnifiedHandState.from_probe(probe)
    assert state.detected and state.is_pointing and state.handedness == "Right"
    assert state.pixel_landmarks.shape == state.world_landmarks_m.shape == (21, 3)
