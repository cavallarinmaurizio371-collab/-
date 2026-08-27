import numpy as np

from src.experimental_3d_pointing.orientation import OrientationEstimate, OrientationSelector


def _estimate(name, valid=True, agreement=5.0):
    return OrientationEstimate(name, valid, "VALID" if valid else "INVALID",
        rotation=np.eye(3) if valid else None,
        projection_agreement={"C": {"mcp_tip_deg": agreement, "pip_tip_deg": agreement}})


def test_backend_prefers_valid_palm_without_reading_target_labels():
    selector = OrientationSelector({"max_projection_agreement_deg": 35,
        "max_rotation_jump_deg": 30,
        "priority": ["R_PALM_STABLE", "R21_RANSAC", "R21_CURRENT"]})
    name, _, quality = selector.select({
        "R_PALM_STABLE": _estimate("R_PALM_STABLE"),
        "R21_RANSAC": _estimate("R21_RANSAC"),
        "R21_CURRENT": _estimate("R21_CURRENT")}, "C")
    assert name == "R_PALM_STABLE" and quality == "GOOD"
