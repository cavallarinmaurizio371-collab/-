import numpy as np

from src.experimental_3d_pointing.orientation import projection_agreement
from tests.evaluation._mentor_helpers import INTRINSICS


def test_projection_agreement_can_flag_extra_rotation_as_worse_without_gt():
    image = np.zeros((21, 2), dtype=float)
    image[5] = [220, 240]; image[6] = [260, 240]; image[8] = [340, 240]
    raw = projection_agreement([1, 0, -1], image, INTRINSICS)
    rotated = projection_agreement([0, 1, -1], image, INTRINSICS)
    assert raw["mcp_tip_deg"] < rotated["mcp_tip_deg"]
