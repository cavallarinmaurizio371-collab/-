from src.experimental_3d_pointing.generic_detector import resolve_object_class_configuration


def test_all_mode_enables_every_supported_class():
    supported = ("cup", "bottle", "book")
    result = resolve_object_class_configuration(supported, True, "ALL", ["person"])
    assert result.all_classes and result.included_classes == supported
