from types import SimpleNamespace

from evaluation.phase2b_metrics import select_quality_fallback


def test_fallback_uses_quality_not_ground_truth():
    valid=SimpleNamespace(in_target_range=True,complete_ray_valid=True)
    name,hit=select_quality_fallback(valid,"GOOD",valid,"GOOD")
    assert name=="C_AXIS_FIT" and hit is valid
    name,hit=select_quality_fallback(valid,"HAND_AXIS_UNSTABLE",valid,"GOOD")
    assert name=="B_RELATIVE" and hit is valid

