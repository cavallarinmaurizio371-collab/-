from src.experimental_3d_pointing.orientation import projection_agreement
from tests.evaluation._mentor_helpers import INTRINSICS, synthetic_correspondences


def test_projected_direction_agrees_with_visible_finger_axis():
    world, image, rotation = synthetic_correspondences()
    direction = rotation @ (world[8] - world[5])
    result = projection_agreement(direction, image, INTRINSICS)
    assert result["mcp_tip_deg"] < 1.0
