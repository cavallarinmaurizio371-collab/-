import numpy as np

from evaluation.diagnostics import assess_direction
from evaluation.z0_geometry import validate_hit_range


CONFIG = {
    "minimum_direction_norm_m": 0.005,
    "near_parallel_abs_dz": 0.15,
    "good_min_toward_abs_dz": 0.55,
    "depth_order_tolerance_m": 0.005,
}


def test_tip_closer_produces_negative_dz_and_good_direction():
    pip=np.array([0.0,0.0,0.70]); tip=np.array([0.0,0.0,0.60])
    direction=(tip-pip)/np.linalg.norm(tip-pip)
    diagnostic=assess_direction(pip,tip,direction,CONFIG)
    assert direction[2] < 0
    assert diagnostic.quality == "GOOD"
    assert diagnostic.depth_order_status == "EXPECTED_TIP_CLOSER"
    assert not diagnostic.sanity_flags


def test_tip_farther_is_reported_as_depth_order_inconsistent_without_reversal():
    pip=np.array([0.0,0.0,0.60]); tip=np.array([0.0,0.0,0.70])
    direction=(tip-pip)/np.linalg.norm(tip-pip)
    diagnostic=assess_direction(pip,tip,direction,CONFIG)
    assert diagnostic.quality == "AWAY"
    assert "DEPTH_ORDER_INCONSISTENT" in diagnostic.sanity_flags
    assert np.allclose(diagnostic.normalized_direction,direction)


def test_direction_sign_mismatch_is_flagged():
    diagnostic=assess_direction([0,0,.7],[0,0,.6],[0,0,1],CONFIG)
    assert "DIRECTION_SIGN_INCONSISTENT" in diagnostic.sanity_flags


def test_a4_range_and_nonfinite_protection():
    assert validate_hit_range([105,148.5],210,297) == "VALID"
    assert validate_hit_range([106,0],210,297) == "OUT_OF_TARGET_RANGE"
    assert validate_hit_range([np.inf,0],210,297) == "NON_FINITE_INTERSECTION"
