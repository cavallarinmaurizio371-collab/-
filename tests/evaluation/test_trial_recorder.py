from types import SimpleNamespace

import numpy as np

from evaluation.trial_recorder import ActiveTrial, TrialRecorder
from src.safety.path_guard import PROJECT_ROOT


def _frame(index):
    return SimpleNamespace(
        timestamp=f"2026-01-01T00:00:0{index}", hand_detected=True,
        gesture_label="POINTING", gesture_confidence=.9,
        points_2d={name:np.array([100+index,200]) for name in ("mcp","pip","dip","tip")},
        points_3d={name:np.array([0,0,.72-offset-index*.01])
                   for name,offset in {"mcp":0,"pip":.01,"dip":.06,"tip":.11}.items()},
        baseline_direction=np.array([0,0,-1.0]), ray_valid=True,
        intersection_valid=True, intersection_status="VALID",
        hit_eval_mm=np.array([index-1.0,0]), pred_region="CENTER", fps=10.0,
        depth_mode="METRIC_RAW_UNCALIBRATED", intrinsics_mode="APPROXIMATE_INTRINSICS")


def test_trial_recorder_writes_complete_artifacts():
    targets={"CENTER":(0.0,0.0)}
    context={"trial_id":"trial_test","gt_target":"CENTER","gt_xy":targets["CENTER"],
             "distance_cm":70,"hand_position":"LEFT","mirror_display":True}
    active=ActiveTrial(context,[_frame(0),_frame(1),_frame(2)])
    recorder=TrialRecorder(PROJECT_ROOT/"outputs/z0_eval/test_artifacts")
    trial=recorder.save_trial(active,targets,{},3)
    recorder.close("PASS")
    assert trial["intersection_valid"] is True
    assert trial["region_correct"] is True
    for name in ("trials.csv","frames.csv","summary.json","summary.md","failure_analysis.md",
                 "region_confusion_matrix.csv","region_confusion_matrix.png"):
        assert (recorder.session_dir/name).exists()
    assert trial["raw_frame_direction"]
    assert trial["median_direction"]
    assert trial["stable_frame_direction"]
    assert trial["failure_reason"] == "VALID_CORRECT_REGION"
