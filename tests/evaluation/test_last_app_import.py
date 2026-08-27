import importlib


def test_last_app_import_has_main_without_starting_camera():
    module = importlib.import_module("last_app")
    assert callable(module.main)
