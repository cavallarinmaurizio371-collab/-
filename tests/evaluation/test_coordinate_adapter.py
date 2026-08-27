import numpy as np

from evaluation.coordinate_adapter import camera_hit_to_eval_mm, mirror_display_x, nearest_region


def test_camera_to_user_facing_eval_axes():
    assert np.allclose(camera_hit_to_eval_mm([.1,.2,0]),[-100,-200])


def test_display_mirror_does_not_change_internal_region():
    targets={"LEFT":(-100,0),"CENTER":(0,0),"RIGHT":(100,0)}
    hit=(-80,5)
    before=nearest_region(*hit,targets)
    assert mirror_display_x(10,640,False)==10
    assert mirror_display_x(10,640,True)==629
    after=nearest_region(*hit,targets)
    assert before==after=="LEFT"

