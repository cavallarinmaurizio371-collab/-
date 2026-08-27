from pathlib import Path


def test_last_app_and_generic_detector_have_no_cup_only_label_gate():
    root = Path(__file__).resolve().parents[2]
    source = (root / "last_app.py").read_text(encoding="utf-8") + \
             (root / "src/experimental_3d_pointing/generic_detector.py").read_text(encoding="utf-8")
    assert "CUP_LABEL" not in source
    assert 'class_name != "cup"' not in source
