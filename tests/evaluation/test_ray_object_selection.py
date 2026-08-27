import numpy as np

from src.experimental_3d_pointing.generic_detector import ObjectDetection
from src.experimental_3d_pointing.object_selection import select_object_by_ray
from tests.evaluation._mentor_helpers import INTRINSICS


def test_best_ray_geometry_selects_matching_configured_object():
    bottle = ObjectDetection(1, 44, "bottle", .8, (300, 220, 340, 260), (320, 240),
        center_camera=np.array([0, 0, 1.2]), depth_valid=True)
    book = ObjectDetection(2, 84, "book", .95, (500, 220, 560, 280), (530, 250),
        center_camera=np.array([.25, 0, .9]), depth_valid=True)
    selected, metric, _ = select_object_by_ray([book, bottle], [0, 0, .5], [0, 0, 1], INTRINSICS)
    assert selected.class_name == "bottle" and metric.angle_deg == 0
