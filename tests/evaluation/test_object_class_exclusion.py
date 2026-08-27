from src.experimental_3d_pointing.generic_detector import GenericObjectDetector, ObjectClassConfiguration


def test_excluded_class_is_rejected_even_in_all_mode():
    detector = GenericObjectDetector.__new__(GenericObjectDetector)
    detector.included = set(); detector.excluded = {"person"}
    detector.class_config = ObjectClassConfiguration(True, True, ("person", "cup"), ("person",), ())
    assert not detector.class_allowed(1, "person")
    assert detector.class_allowed(47, "cup")
