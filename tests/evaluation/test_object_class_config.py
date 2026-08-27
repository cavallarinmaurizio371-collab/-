from src.experimental_3d_pointing.generic_detector import resolve_object_class_configuration


def test_specific_runtime_target_classes_and_unknown_warning():
    result = resolve_object_class_configuration(("cup", "bottle", "book"), True,
                                                ["cup", "book", "spaceship"], [])
    assert result.enabled and result.included_classes == ("cup", "book")
    assert result.unknown_classes == ("spaceship",)


def test_empty_target_class_list_disables_object_selection():
    result = resolve_object_class_configuration(("cup", "book"), True, [], [])
    assert not result.enabled
