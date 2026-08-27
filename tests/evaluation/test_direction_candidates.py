from types import SimpleNamespace

import cv2
import numpy as np

from evaluation.direction_candidates import (build_candidates,
    estimate_hand_to_camera_rotation,fit_finger_axis)


INTRINSICS=SimpleNamespace(fx=800.0,fy=800.0,cx=320.0,cy=240.0,
                           distortion=[0,0,0,0,0])


def _synthetic_hand():
    rng=np.random.default_rng(7)
    world=rng.normal(0,.025,(21,3))
    world[5]=[-.01,.02,.01]
    world[6]=[-.005,.005,-.01]
    world[7]=[0,-.01,-.03]
    world[8]=[.005,-.025,-.055]
    rvec=np.array([.08,-.12,.04]); tvec=np.array([0.02,-.01,.65])
    matrix=np.array([[800,0,320],[0,800,240],[0,0,1]],float)
    image,_=cv2.projectPoints(world,rvec,tvec,matrix,np.zeros(5))
    rotation,_=cv2.Rodrigues(rvec)
    return world,image.reshape(-1,2),rotation


def test_axis_fit_is_oriented_from_mcp_to_tip():
    world,_,_=_synthetic_hand(); axis=fit_finger_axis(world)
    assert np.dot(axis,world[8]-world[5]) > 0
    assert np.isclose(np.linalg.norm(axis),1)


def test_pnp_recovers_missing_hand_to_camera_rotation():
    world,image,expected= _synthetic_hand()
    mapping=estimate_hand_to_camera_rotation(world,image,INTRINSICS)
    assert mapping.valid
    assert mapping.reprojection_rmse_px < 1e-3
    assert np.allclose(mapping.rotation,expected,atol=1e-3)


def test_pnp_accepts_live_probe_21x3_pixel_array():
    world,image,_=_synthetic_hand()
    live_shape=np.column_stack([image,np.zeros(21)])
    mapping=estimate_hand_to_camera_rotation(world,live_shape,INTRINSICS)
    assert mapping.valid
    assert mapping.reprojection_rmse_px < 1e-3


def test_all_three_candidates_are_kept_separate():
    world,image,rotation=_synthetic_hand()
    scene_pip=np.array([0,0,.7]); scene_tip=np.array([0,0,.6])
    candidates,mapping=build_candidates(scene_pip,scene_tip,world,image,INTRINSICS)
    assert mapping.valid
    assert candidates["A_BASELINE"].camera_status == "TOWARD"
    expected=rotation@((world[8]-world[6])/np.linalg.norm(world[8]-world[6]))
    assert np.allclose(candidates["B_RELATIVE"].camera_direction,expected,atol=1e-3)
    assert candidates["C_AXIS_FIT"].camera_direction is not None
