import numpy as np

from evaluation.eval_pipeline import EvalFrameResult
from evaluation.visualization import (direction_display_endpoints,
                                      draw_hand_overlay, write_png)
from src.safety.path_guard import PROJECT_ROOT


def _result(status="VALID", intersection_valid=True):
    landmarks=np.array([[60+(i%5)*20,70+(i//5)*25,0] for i in range(21)],dtype=float)
    result=EvalFrameResult("now",True,"POINTING",.9)
    result.landmarks_2d=landmarks
    result.direction_2d=np.array([1.0,0.0])
    result.points_2d={"mcp":landmarks[5,:2],"pip":landmarks[6,:2],
                      "dip":landmarks[7,:2],"tip":landmarks[8,:2]}
    result.ray_valid=True
    result.intersection_valid=intersection_valid
    result.intersection_status=status
    return result


def test_complete_skeleton_and_valid_direction_are_drawn():
    image=np.zeros((300,400,3),dtype=np.uint8)
    visible=draw_hand_overlay(image,_result(),False,400,120)
    assert visible
    assert np.count_nonzero(image[:,:,1])>100
    assert np.count_nonzero(image[:,:,2])>20


def test_invalid_intersection_keeps_current_direction_visible():
    image=np.zeros((300,400,3),dtype=np.uint8)
    visible=draw_hand_overlay(
        image,_result("POINTING_AWAY_FROM_CAMERA",False),False,400,120)
    assert visible
    # Orange BGR line has strong green and red components.
    assert np.any((image[:,:,1]>120)&(image[:,:,2]>200))


def test_mirror_flips_both_direction_endpoints_only_for_display():
    start,end=direction_display_endpoints([100,120],[1,0],400,False,80)
    mirrored_start,mirrored_end=direction_display_endpoints([100,120],[1,0],400,True,80)
    assert start==(100,120) and end==(180,120)
    assert mirrored_start==(299,120) and mirrored_end==(219,120)


def test_unicode_safe_visualization_screenshot_write():
    image=np.zeros((20,20,3),dtype=np.uint8)
    path=PROJECT_ROOT/"outputs/z0_eval/visualization_write_test.png"
    assert write_png(path,image)
    assert path.exists() and path.stat().st_size>0
