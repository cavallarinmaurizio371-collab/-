from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.camera.calibration import CameraIntrinsics,load_intrinsics


def load_phase2_intrinsics(project_root,config,width,height):
    calibrated=project_root/config["intrinsics"]["calibrated_file"]
    if config["intrinsics"].get("prefer_calibrated",True) and calibrated.exists():
        try:
            data=json.loads(calibrated.read_text(encoding="utf-8"))
            distortion=data.get("distortion",data.get("dist_coeffs",[0.0]*5))
            intrinsics=CameraIntrinsics(float(data["fx"]),float(data["fy"]),
                float(data["cx"]),float(data["cy"]),list(distortion),
                int(data["image_width"]),int(data["image_height"]),
                bool(data.get("valid_calibration",True)),"CALIBRATED_INTRINSICS")
            if intrinsics.valid_calibration:
                return intrinsics.scaled(width,height)
        except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):
            pass
    fallback=project_root/config["intrinsics"]["fallback_file"]
    intrinsics=load_intrinsics(fallback,width,height)
    intrinsics.mode="APPROXIMATE_INTRINSICS_FALLBACK"
    return intrinsics


def camera_matrix(intrinsics):
    return np.asarray([[intrinsics.fx,0.0,intrinsics.cx],
                       [0.0,intrinsics.fy,intrinsics.cy],
                       [0.0,0.0,1.0]],dtype=np.float64)


def distortion_coefficients(intrinsics):
    return np.asarray(intrinsics.distortion,dtype=np.float64).reshape(-1)


def backproject_distorted_pixel(u,v,depth_m,intrinsics):
    """Back-project a raw camera pixel using the same distortion model as PnP."""
    depth=float(depth_m)
    if not np.isfinite(depth) or depth<=0:
        raise ValueError("Depth must be finite and positive")
    pixel=np.ascontiguousarray([[[float(u),float(v)]]],dtype=np.float64)
    normalized=cv2.undistortPoints(pixel,camera_matrix(intrinsics),
                                   distortion_coefficients(intrinsics)).reshape(2)
    value=np.asarray([normalized[0]*depth,normalized[1]*depth,depth],dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("Back-projection produced a non-finite point")
    return value


def project_camera_points(points_camera,intrinsics):
    """Project Camera-space points to raw distorted image pixels."""
    points=np.asarray(points_camera,dtype=np.float64).reshape(-1,3)
    if not len(points) or not np.all(np.isfinite(points)) or np.any(points[:,2]<=0):
        raise ValueError("Projection requires finite points with positive Camera Z")
    pixels,_=cv2.projectPoints(np.ascontiguousarray(points),np.zeros(3),np.zeros(3),
                               camera_matrix(intrinsics),distortion_coefficients(intrinsics))
    return pixels.reshape(-1,2)


def load_phase2b_intrinsics(project_root,config,width,height):
    """Strict formal-evaluation loader: no approximate fallback and no scaling."""
    path=Path(project_root)/config["intrinsics"]["calibrated_file"]
    if not path.exists():
        raise RuntimeError(f"CALIBRATED_INTRINSICS_REQUIRED: {path}")
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
        intrinsics=CameraIntrinsics(float(data["fx"]),float(data["fy"]),
            float(data["cx"]),float(data["cy"]),
            list(data.get("distortion",data["dist_coeffs"])),
            int(data["image_width"]),int(data["image_height"]),
            bool(data.get("valid_calibration",False)),"CALIBRATED_INTRINSICS")
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
        raise RuntimeError(f"INVALID_CALIBRATED_INTRINSICS: {path}") from error
    if not intrinsics.valid_calibration:
        raise RuntimeError("CALIBRATED_INTRINSICS_REQUIRED: calibration is not valid")
    if (int(width),int(height))!=(intrinsics.image_width,intrinsics.image_height):
        raise RuntimeError("CALIBRATION_RESOLUTION_MISMATCH: "
                           f"camera={width}x{height}, calibration="
                           f"{intrinsics.image_width}x{intrinsics.image_height}")
    return intrinsics
