from src.experimental_3d_pointing.generic_detector import GenericObjectDetector


def test_default_generic_filter_accepts_multiple_non_cup_classes():
    detector = GenericObjectDetector.__new__(GenericObjectDetector)
    detector.included = set(); detector.excluded = set()
    assert detector.class_allowed(44, "bottle")
    assert detector.class_allowed(73, "laptop")
    assert detector.class_allowed(47, "cup")
