from types import SimpleNamespace

from src.experimental_3d_pointing.coarse import CoarseTemporalStabilizer


def test_region_requires_stable_frames_and_holds_briefly():
    temporal = CoarseTemporalStabilizer(window_size=5, min_stable_frames=3,
                                        hold_frames=2, max_jitter_deg=2)
    hit = SimpleNamespace(valid=True, yaw_deg=8.0, pitch_deg=0.0)
    assert temporal.update(hit, True)[1] is None
    assert temporal.update(hit, True)[1] is None
    assert temporal.update(hit, True)[1] == "RIGHT"
    assert temporal.update(None, False)[0] == "HOLDING"
