from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DirectionDiagnostic:
    quality: str
    ray_status: str
    raw_direction: np.ndarray | None = None
    direction_norm: float | None = None
    normalized_direction: np.ndarray | None = None
    angle_to_camera_axis_deg: float | None = None
    depth_order_status: str = "MISSING_KEYPOINT"
    sanity_flags: list[str] = field(default_factory=list)


def assess_direction(pip_xyz, tip_xyz, normalized_direction, config) -> DirectionDiagnostic:
    if pip_xyz is None or tip_xyz is None or normalized_direction is None:
        return DirectionDiagnostic("INVALID", "INVALID")
    pip=np.asarray(pip_xyz,dtype=float); tip=np.asarray(tip_xyz,dtype=float)
    direction=np.asarray(normalized_direction,dtype=float)
    if not (np.all(np.isfinite(pip)) and np.all(np.isfinite(tip)) and
            np.all(np.isfinite(direction))):
        return DirectionDiagnostic("INVALID","INVALID",sanity_flags=["NON_FINITE_DIRECTION"])
    raw=tip-pip; norm=float(np.linalg.norm(raw))
    minimum=float(config["minimum_direction_norm_m"])
    if norm<minimum or np.linalg.norm(direction)<1e-9:
        return DirectionDiagnostic("INVALID","INVALID",raw,norm,sanity_flags=["DIRECTION_TOO_SHORT"])
    direction=direction/np.linalg.norm(direction)
    dz=float(direction[2]); tolerance=float(config["depth_order_tolerance_m"])
    delta_z=float(tip[2]-pip[2])
    if delta_z < -tolerance:
        depth_status="EXPECTED_TIP_CLOSER"
    elif delta_z > tolerance:
        depth_status="DEPTH_ORDER_INCONSISTENT"
    else:
        depth_status="DEPTH_ORDER_AMBIGUOUS"
    flags=[]
    raw_sign=np.sign(delta_z) if abs(delta_z)>tolerance else 0
    direction_sign=np.sign(dz) if abs(dz)>1e-9 else 0
    if raw_sign and direction_sign and raw_sign!=direction_sign:
        flags.append("DIRECTION_SIGN_INCONSISTENT")
    if depth_status=="DEPTH_ORDER_INCONSISTENT":
        flags.append(depth_status)
    camera_axis=np.array([0.0,0.0,-1.0])
    angle=float(np.degrees(np.arccos(np.clip(np.dot(direction,camera_axis),-1.0,1.0))))
    if dz>0:
        quality,ray_status="AWAY","AWAY_FROM_CAMERA"
    elif abs(dz)<float(config["near_parallel_abs_dz"]):
        quality,ray_status="NEAR_PARALLEL","NEAR_PARALLEL"
    elif abs(dz)>=float(config["good_min_toward_abs_dz"]):
        quality,ray_status="GOOD","TOWARD_CAMERA"
    else:
        quality,ray_status="MARGINAL","TOWARD_CAMERA"
    return DirectionDiagnostic(quality,ray_status,raw,norm,direction,angle,depth_status,flags)


def classify_keypoint_stability(stability, config):
    pixel_values=[stability.get(f"{name}_pixel_std") for name in ("tip","dip","pip")]
    xyz_values=[stability.get(f"{name}_xyz_std_m") for name in ("tip","dip","pip")]
    pixel=max((v for v in pixel_values if v is not None),default=None)
    xyz=max((v for v in xyz_values if v is not None),default=None)
    if pixel is None or xyz is None:
        return "BAD"
    p_limit=float(config.get("keypoint_pixel_std_threshold",18.0))
    x_limit=float(config.get("keypoint_xyz_std_m_threshold",0.08))
    if pixel<=p_limit*.5 and xyz<=x_limit*.5:
        return "GOOD"
    if pixel<=p_limit and xyz<=x_limit:
        return "MEDIUM"
    return "BAD"
