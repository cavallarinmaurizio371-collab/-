from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_empty_or_invalid_object_set_returns_none():
    selected, metric, candidates = select_object_by_ray([], [0, 0, .5], [0, 0, 1], INTRINSICS)
    assert selected is None and metric is None and candidates == []
