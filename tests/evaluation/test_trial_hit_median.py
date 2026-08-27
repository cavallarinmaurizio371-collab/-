import numpy as np

from evaluation.phase2b_metrics import trial_hit_median


def test_trial_prediction_uses_median_and_keeps_outlier_in_jitter():
    prediction,jitter=trial_hit_median([[0,0],[1,1],[2,2],[100,100]],3)
    assert np.allclose(prediction,[1.5,1.5])
    assert jitter>40

