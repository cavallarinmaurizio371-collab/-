from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8


@dataclass(frozen=True)
class CameraMapping:
    valid: bool
    status: str
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    reprojection_rmse_px: float | None = None


@dataclass(frozen=True)
class DirectionCandidate:
    name: str
    native_direction: np.ndarray | None
    camera_direction: np.ndarray | None
    camera_status: str
    mapping_status: str


@dataclass(frozen=True)
class AxisFitResult:
    valid: bool
    direction: np.ndarray | None = None
    residual_m: float | None = None
    linearity: float | None = None
    status: str = "INVALID"


def _points(value, expected_rows=None):
    array=np.asarray(value,dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("Landmarks must be a finite Nx3 array")
    if expected_rows is not None and len(array) != expected_rows:
        raise ValueError(f"Expected {expected_rows} landmarks")
    return array


def _normalize(vector):
    value=np.asarray(vector,dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return None
    norm=float(np.linalg.norm(value))
    return value/norm if norm > 1e-9 else None


def classify_camera_direction(direction, near_parallel_abs_dz=.15):
    value=_normalize(direction)
    if value is None:
        return "UNKNOWN"
    if value[2] > 0:
        return "AWAY"
    if abs(float(value[2])) < float(near_parallel_abs_dz):
        return "NEAR_PARALLEL"
    return "TOWARD"


def fit_finger_axis_with_quality(world_landmarks):
    """TLS/PCA axis through MCP/PIP/DIP/TIP, oriented from MCP toward TIP."""
    try:
        world=_points(world_landmarks,21)
        finger=world[[INDEX_MCP,INDEX_PIP,INDEX_DIP,INDEX_TIP]]
        centered=finger-finger.mean(axis=0)
        _,singular,vh=np.linalg.svd(centered,full_matrices=False)
        axis=_normalize(vh[0])
        if axis is None:
            return AxisFitResult(False,status="DEGENERATE_AXIS")
        orientation=world[INDEX_TIP]-world[INDEX_MCP]
        if np.dot(axis,orientation) < 0:
            axis=-axis
        projected=np.outer(centered@axis,axis)
        residual=float(np.sqrt(np.mean(np.sum((centered-projected)**2,axis=1))))
        energy=singular*singular
        linearity=float(energy[0]/energy.sum()) if energy.sum()>1e-12 else 0.0
        return AxisFitResult(True,axis,residual,linearity,"VALID")
    except (ValueError,TypeError,np.linalg.LinAlgError):
        return AxisFitResult(False,status="INVALID_AXIS_INPUT")


def fit_finger_axis(world_landmarks):
    return fit_finger_axis_with_quality(world_landmarks).direction


def estimate_hand_to_camera_rotation(world_landmarks, image_landmarks_px, intrinsics,
                                     max_reprojection_rmse_px=25.0):
    """Estimate R_camera_from_hand using 3D/2D landmark correspondences.

    MediaPipe documents the world points' scale and hand-centred origin, but its
    Python result does not provide a camera extrinsic. PnP supplies the missing
    per-frame rotation. Approximate intrinsics make this diagnostic, not a
    calibrated production transform.
    """
    try:
        world=np.ascontiguousarray(_points(world_landmarks,21),dtype=np.float64)
        image=np.asarray(image_landmarks_px,dtype=np.float64)
        if image.shape not in ((21,2),(21,3)) or not np.all(np.isfinite(image)):
            raise ValueError("Image landmarks must be finite 21x2/21x3")
        # A live probe supplies 21x3 (pixel x/y plus relative z). Slicing the
        # first two columns is non-contiguous; OpenCV solvePnP requires a
        # contiguous point matrix on Windows.
        image=np.ascontiguousarray(image[:,:2],dtype=np.float64)
        camera_matrix=np.array([[intrinsics.fx,0,intrinsics.cx],
                                [0,intrinsics.fy,intrinsics.cy],
                                [0,0,1]],dtype=np.float64)
        distortion=np.asarray(intrinsics.distortion,dtype=np.float64)
        success,rvec,tvec=cv2.solvePnP(world,image,camera_matrix,distortion,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            return CameraMapping(False,"PNP_FAILED")
        rotation,_=cv2.Rodrigues(rvec)
        projected,_=cv2.projectPoints(world,rvec,tvec,camera_matrix,distortion)
        residual=projected.reshape(-1,2)-image
        rmse=float(np.sqrt(np.mean(np.sum(residual*residual,axis=1))))
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(tvec)):
            return CameraMapping(False,"PNP_NON_FINITE",reprojection_rmse_px=rmse)
        if float(tvec.reshape(3)[2]) <= 0:
            return CameraMapping(False,"PNP_BEHIND_CAMERA",reprojection_rmse_px=rmse)
        if rmse > float(max_reprojection_rmse_px):
            return CameraMapping(False,"PNP_HIGH_REPROJECTION_ERROR",
                                 reprojection_rmse_px=rmse)
        mode=getattr(intrinsics,"mode","APPROXIMATE_INTRINSICS")
        status=("PNP_CALIBRATED_INTRINSICS" if mode=="CALIBRATED_INTRINSICS"
                else "PNP_APPROXIMATE_INTRINSICS")
        return CameraMapping(True,status,rotation,
                             tvec.reshape(3),rmse)
    except (ValueError,cv2.error,np.linalg.LinAlgError):
        return CameraMapping(False,"PNP_INVALID_INPUT")


def build_candidates(scene_pip_xyz, scene_tip_xyz, world_landmarks,
                     image_landmarks_px, intrinsics, near_parallel_abs_dz=.15):
    baseline=_normalize(np.asarray(scene_tip_xyz,dtype=float)-np.asarray(scene_pip_xyz,dtype=float)) \
        if scene_pip_xyz is not None and scene_tip_xyz is not None else None
    world=None
    try:
        world=_points(world_landmarks,21)
    except (ValueError,TypeError):
        pass
    relative=_normalize(world[INDEX_TIP]-world[INDEX_PIP]) if world is not None else None
    axis=fit_finger_axis(world) if world is not None else None
    mapping=(estimate_hand_to_camera_rotation(world,image_landmarks_px,intrinsics)
             if world is not None else CameraMapping(False,"WORLD_LANDMARKS_UNAVAILABLE"))

    def mapped(name,native):
        camera=_normalize(mapping.rotation@native) if mapping.valid and native is not None else None
        return DirectionCandidate(name,native,camera,
                                  classify_camera_direction(camera,near_parallel_abs_dz),
                                  mapping.status)
    return {
        "A_BASELINE":DirectionCandidate("A_BASELINE",baseline,baseline,
                                        classify_camera_direction(baseline,near_parallel_abs_dz),
                                        "SCENE_DEPTH_CAMERA_XYZ"),
        "B_RELATIVE":mapped("B_RELATIVE",relative),
        "C_AXIS_FIT":mapped("C_AXIS_FIT",axis),
    },mapping
