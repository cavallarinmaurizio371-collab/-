from types import SimpleNamespace

import numpy as np

from evaluation.metrics import build_summary, confusion_matrix
from evaluation.stability_analyzer import analyze_stability


def _frame(offset):
    return SimpleNamespace(
        points_2d={name:np.array([100+offset,200]) for name in ("tip","pip","dip")},
        points_3d={name:np.array([0.01*offset,0,0.7]) for name in ("tip","pip","dip")},
        baseline_direction=np.array([0,0,-1.0]), hit_eval_mm=np.array([offset,0.0]))


def test_stability_is_computed_from_continuous_frames():
    result=analyze_stability([_frame(-1),_frame(0),_frame(1)],{})
    assert result["tip_pixel_std"]>0
    assert result["direction_angle_std_deg"]==0
    assert result["direction_dz_negative_rate"]==1
    assert result["hit_radial_std_mm"]>0


def test_summary_and_confusion_matrix():
    trial={"gt_target":"CENTER","pred_region":"CENTER","region_correct":True,
           "radial_error_mm":10.0,"intersection_valid":True,
           "measured_hand_distance_cm":70,"hand_position_in_frame":"LEFT"}
    frame={"hand_detected":True,"gesture_label":"POINTING","ray_valid":True,"intersection_valid":True}
    summary=build_summary([trial],[frame],"PASS")
    assert summary["region_accuracy"]==1
    assert summary["mean_radial_error_mm"]==10
    assert confusion_matrix([trial])["CENTER"]["CENTER"]==1

