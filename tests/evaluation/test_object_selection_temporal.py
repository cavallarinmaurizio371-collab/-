from src.experimental_3d_pointing.object_selection import ObjectSelectionHysteresis


def test_object_requires_stable_frames_and_releases_after_hold():
    temporal = ObjectSelectionHysteresis(switch_confirm_frames=3, release_frames=2)
    assert temporal.update(7) is None
    assert temporal.update(7) is None
    assert temporal.update(7) == 7
    assert temporal.update(None) == 7
    assert temporal.update(None) is None
