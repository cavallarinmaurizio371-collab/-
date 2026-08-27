from src.experimental_3d_pointing.intrinsics import load_phase2_intrinsics
from src.safety.path_guard import PROJECT_ROOT


def test_missing_calibration_is_explicit_fallback():
    config={"intrinsics":{"prefer_calibrated":True,
        "calibrated_file":"cache/test_phase2a5/does_not_exist.json",
        "fallback_file":"configs/camera_intrinsics.json"}}
    result=load_phase2_intrinsics(PROJECT_ROOT,config,640,480)
    assert result.mode=="APPROXIMATE_INTRINSICS_FALLBACK"
    assert not result.valid_calibration

