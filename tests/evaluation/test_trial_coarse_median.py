from types import SimpleNamespace

from src.experimental_3d_pointing.coarse import trial_coarse_median


def test_trial_prediction_uses_median_angles():
    hits = [SimpleNamespace(valid=True, yaw_deg=value, pitch_deg=0.0)
            for value in (7.0, 8.0, 60.0)]
    result = trial_coarse_median(hits)
    assert result.region == "RIGHT" and result.yaw_deg == 8.0
