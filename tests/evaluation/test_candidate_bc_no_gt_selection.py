from types import SimpleNamespace

from src.experimental_3d_pointing.coarse import select_candidate_no_gt


def test_c_primary_and_b_fallback_depend_only_on_quality():
    c = SimpleNamespace(valid=True, region="LEFT")
    b = SimpleNamespace(valid=True, region="RIGHT")
    name, chosen = select_candidate_no_gt(c, b, "GOOD", "GOOD", True)
    assert name == "C" and chosen is c
    name, chosen = select_candidate_no_gt(c, b, "HAND_AXIS_UNSTABLE", "GOOD", False)
    assert name == "B_FALLBACK" and chosen is b
