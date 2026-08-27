from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.coordinate_adapter import camera_hit_to_eval_mm,nearest_region
from evaluation.z0_geometry import intersect_ray_with_z0,validate_hit_range


CANDIDATES=("A_BASELINE","B_RELATIVE","C_AXIS_FIT")


@dataclass(frozen=True)
class CandidateHit:
    candidate: str
    complete_ray_valid: bool=False
    toward_camera: bool=False
    direction_quality_valid: bool=False
    intersection_valid: bool=False
    in_target_range: bool=False
    status: str="INVALID"
    hit_eval_mm: np.ndarray | None=None
    pred_region: str | None=None
    error_mm: float | None=None


def _finite_vector(value):
    if value is None:
        return None
    array=np.asarray(value,dtype=float)
    return array if array.shape==(3,) and np.all(np.isfinite(array)) else None


def evaluate_camera_candidate(name,candidate,targets,gt_xy,width_mm,height_mm,epsilon=1e-6):
    origin=_finite_vector(getattr(candidate,"origin_camera",None))
    direction=_finite_vector(getattr(candidate,"smoothed_direction_camera",None))
    quality=str(getattr(candidate,"quality","INVALID"))
    complete=origin is not None and direction is not None and np.linalg.norm(direction)>1e-9
    toward=bool(complete and direction[2]<0)
    quality_valid=quality in ("GOOD","MARGINAL")
    if not complete:
        return CandidateHit(name,status="NUMERICAL_INVALID" if origin is not None or direction is not None
                            else quality)
    intersection=intersect_ray_with_z0(origin,direction,epsilon)
    status=("AWAY_FROM_CAMERA" if intersection.status=="POINTING_AWAY_FROM_CAMERA"
            else intersection.status)
    if not intersection.valid:
        return CandidateHit(name,True,toward,quality_valid,status=status)
    hit=camera_hit_to_eval_mm(intersection.point_camera)
    if not np.all(np.isfinite(hit)):
        return CandidateHit(name,True,toward,quality_valid,status="NUMERICAL_INVALID")
    range_status=validate_hit_range(hit,width_mm,height_mm)
    if range_status!="VALID":
        return CandidateHit(name,True,toward,quality_valid,True,False,
                            "OUT_OF_TARGET_RANGE",hit)
    region=nearest_region(*hit,targets)
    error=float(np.linalg.norm(hit-np.asarray(gt_xy,dtype=float))) if gt_xy is not None else None
    return CandidateHit(name,True,toward,quality_valid,True,True,"VALID",hit,region,error)


def select_quality_fallback(c_hit,c_quality,b_hit,b_quality,c_accepted=("GOOD",),b_accepted=("GOOD",)):
    """Select C then B using only model quality/validity; GT is deliberately absent."""
    if c_quality in c_accepted and c_hit is not None and c_hit.in_target_range:
        return "C_AXIS_FIT",c_hit
    c_invalid=(c_hit is None or not c_hit.complete_ray_valid or
               c_quality in ("INVALID","PNP_UNRELIABLE","GLOBAL_DEPTH_UNSTABLE",
                             "HAND_AXIS_UNSTABLE","AWAY"))
    if c_invalid and b_quality in b_accepted and b_hit is not None and b_hit.in_target_range:
        return "B_RELATIVE",b_hit
    return "INVALID",None


def trial_hit_median(hits,minimum_valid_frames=3):
    points=[np.asarray(point,dtype=float) for point in hits
            if point is not None and np.asarray(point).shape==(2,) and np.all(np.isfinite(point))]
    if len(points)<int(minimum_valid_frames):
        return None,None
    matrix=np.stack(points); median=np.median(matrix,axis=0)
    jitter=float(np.sqrt(np.mean(np.sum((matrix-median)**2,axis=1))))
    return median,jitter


def vector_jitter_deg(vectors):
    values=[]
    for value in vectors:
        vector=_finite_vector(value)
        if vector is not None and np.linalg.norm(vector)>1e-9:
            values.append(vector/np.linalg.norm(vector))
    if not values:
        return None
    matrix=np.stack(values); center=np.mean(matrix,axis=0)
    if np.linalg.norm(center)<=1e-9:
        return None
    center/=np.linalg.norm(center)
    angles=np.degrees(np.arccos(np.clip(matrix@center,-1,1)))
    return float(np.sqrt(np.mean(angles*angles)))


def point_jitter(values):
    points=[_finite_vector(value) for value in values]
    points=[value for value in points if value is not None]
    if not points:
        return None
    matrix=np.stack(points); center=np.median(matrix,axis=0)
    return float(np.sqrt(np.mean(np.sum((matrix-center)**2,axis=1))))


def landing_error(hit,gt_xy):
    if hit is None or gt_xy is None:
        return None
    values=np.asarray(hit,dtype=float)-np.asarray(gt_xy,dtype=float)
    return float(np.linalg.norm(values)) if values.shape==(2,) and np.all(np.isfinite(values)) else None


def region_accuracy(trials,prefix):
    predicted=[row for row in trials if row.get(f"{prefix}_region")]
    return (sum(row[f"{prefix}_region"]==row.get("gt_target") for row in predicted)/len(predicted)
            if predicted else None)


def error_stats(values):
    data=np.asarray([float(value) for value in values
                     if value is not None and np.isfinite(float(value))],dtype=float)
    return {"mean_mm":float(np.mean(data)) if len(data) else None,
            "median_mm":float(np.median(data)) if len(data) else None,
            "p90_mm":float(np.percentile(data,90)) if len(data) else None,
            "max_mm":float(np.max(data)) if len(data) else None}


def paired_candidate_metrics(trials,tie_tolerance_mm=1e-6):
    pairs=[]
    for row in trials:
        b=row.get("B_error_mm"); c=row.get("C_error_mm")
        if b is not None and c is not None and np.isfinite(float(b)) and np.isfinite(float(c)):
            pairs.append(float(c)-float(b))
    b_wins=sum(value>tie_tolerance_mm for value in pairs)
    c_wins=sum(value<-tie_tolerance_mm for value in pairs)
    ties=len(pairs)-b_wins-c_wins
    return {"paired_trials":len(pairs),"B_win_count":b_wins,"C_win_count":c_wins,
            "tie_count":ties,
            "median_C_minus_B_mm":float(np.median(pairs)) if pairs else None,
            "p90_C_minus_B_mm":float(np.percentile(pairs,90)) if pairs else None}


def session_is_excluded(session_id,excluded_sessions):
    return str(session_id) in {str(value) for value in excluded_sessions}
