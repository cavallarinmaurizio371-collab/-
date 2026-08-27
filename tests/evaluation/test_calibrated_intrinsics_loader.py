import json

from src.experimental_3d_pointing.intrinsics import load_phase2_intrinsics
from src.safety.path_guard import PROJECT_ROOT,safe_mkdir,safe_open


def test_calibrated_intrinsics_are_preferred_and_scaled():
    folder=safe_mkdir(PROJECT_ROOT/"cache"/"test_phase2a5")
    path=folder/"calibrated.json"
    payload={"image_width":640,"image_height":480,"fx":600,"fy":610,"cx":320,"cy":240,
             "dist_coeffs":[.1,0,0,0,0],"valid_calibration":True}
    with safe_open(path,"w",encoding="utf-8") as handle: json.dump(payload,handle)
    config={"intrinsics":{"prefer_calibrated":True,
        "calibrated_file":str(path.relative_to(PROJECT_ROOT)),
        "fallback_file":"configs/camera_intrinsics.json"}}
    result=load_phase2_intrinsics(PROJECT_ROOT,config,1280,960)
    assert result.mode=="CALIBRATED_INTRINSICS"
    assert result.fx==1200 and result.fy==1220

