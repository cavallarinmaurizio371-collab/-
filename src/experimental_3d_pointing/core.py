from __future__ import annotations

from dataclasses import dataclass,field

import numpy as np

from evaluation.direction_candidates import (estimate_hand_to_camera_rotation,
    fit_finger_axis_with_quality)
from src.depth.depth_sampler import sample_point
from src.experimental_3d_pointing.intrinsics import backproject_distorted_pixel


@dataclass(frozen=True)
class AnchorDepthResult:
    valid: bool
    median_m: float | None = None
    mad_m: float | None = None
    valid_samples: int = 0
    status: str = "INVALID"
    method: str = "V1_FIVE_POINT_MEDIAN"
    point_depths_m: dict = field(default_factory=dict)
    point_positions_px: dict = field(default_factory=dict)


@dataclass
class RayCandidate:
    name: str
    origin_camera: np.ndarray | None = None
    raw_direction_camera: np.ndarray | None = None
    smoothed_direction_camera: np.ndarray | None = None
    angle_to_camera_deg: float | None = None
    quality: str = "INVALID"
    temporal_angle_jump_deg: float | None = None


@dataclass
class PointingCoreResult:
    candidates: dict[str,RayCandidate]=field(default_factory=dict)
    pnp_status: str = "INVALID"
    pnp_rmse_px: float | None = None
    anchor: AnchorDepthResult=field(default_factory=lambda:AnchorDepthResult(False))
    anchor_v1: AnchorDepthResult=field(default_factory=lambda:AnchorDepthResult(False))
    anchor_v2: AnchorDepthResult=field(default_factory=lambda:AnchorDepthResult(False))
    raw_anchor_depth_m: float | None = None
    filtered_anchor_depth_m: float | None = None
    anchor_temporal_status: str = "INVALID"
    raw_anchor_camera: np.ndarray | None = None
    filtered_anchor_camera: np.ndarray | None = None
    anchor_camera: np.ndarray | None = None
    tip_camera: np.ndarray | None = None
    axis_residual_m: float | None = None
    axis_linearity: float | None = None
    intrinsics_mode: str = "UNKNOWN"
    world_tip_anchor_distance_m: float | None = None
    world_finger_length_m: float | None = None


def _normalize(value):
    vector=np.asarray(value,dtype=float)
    if vector.shape!=(3,) or not np.all(np.isfinite(vector)):
        return None
    norm=float(np.linalg.norm(vector))
    return vector/norm if norm>1e-9 else None


def _angle_to_camera(direction):
    direction=_normalize(direction)
    if direction is None: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(direction,[0,0,-1]),-1,1))))


def robust_anchor_depth(depth_map,pixel_landmarks,indices,patch_size,
                        minimum_samples,max_mad_m,corrector=None):
    values=[]
    if depth_map is not None and pixel_landmarks is not None:
        pixels=np.asarray(pixel_landmarks,dtype=float)
        for index in indices:
            sampled=sample_point(depth_map,pixels[int(index),:2],patch_size)
            if sampled is not None:
                corrected=corrector(sampled) if corrector else sampled
                if corrected is not None and np.isfinite(corrected) and corrected>0:
                    values.append(float(corrected))
    if len(values)<int(minimum_samples):
        return AnchorDepthResult(False,valid_samples=len(values),status="INSUFFICIENT_ANCHOR_DEPTH")
    median=float(np.median(values)); mad=float(np.median(np.abs(np.asarray(values)-median)))
    if mad>float(max_mad_m):
        return AnchorDepthResult(False,median,mad,len(values),"GLOBAL_DEPTH_UNSTABLE")
    return AnchorDepthResult(True,median,mad,len(values),"VALID")


def _patch_values(depth_map,point,patch_size,corrector=None):
    x,y=[int(round(v)) for v in point]; radius=max(1,int(patch_size)//2)
    height,width=depth_map.shape[:2]
    values=np.asarray(depth_map[max(0,y-radius):min(height,y+radius+1),
                                max(0,x-radius):min(width,x+radius+1)],dtype=float).reshape(-1)
    values=values[np.isfinite(values)&(values>0)]
    if corrector:
        values=np.asarray([corrector(value) for value in values],dtype=float)
        values=values[np.isfinite(values)&(values>0)]
    return values


def palm_anchor_candidates(depth_map,pixel_landmarks,indices,patch_size,minimum_samples,
                           max_mad_m,corrector=None,lower_percentile=10,
                           upper_percentile=90,outlier_mad_scale=3.5):
    names=("WRIST","INDEX_MCP","MIDDLE_MCP","RING_MCP","PINKY_MCP")
    pixels=np.asarray(pixel_landmarks,dtype=float) if pixel_landmarks is not None else None
    point_depths={}; positions={}; pooled=[]
    if depth_map is not None and pixels is not None:
        for name,index in zip(names,indices):
            positions[name]=[float(v) for v in pixels[int(index),:2]]
            values=_patch_values(depth_map,pixels[int(index),:2],patch_size,corrector)
            if len(values):
                point_depths[name]=float(np.median(values)); pooled.extend(values.tolist())
    depths=list(point_depths.values())
    if len(depths)>=int(minimum_samples):
        median=float(np.median(depths)); mad=float(np.median(np.abs(np.asarray(depths)-median)))
        v1=AnchorDepthResult(mad<=float(max_mad_m),median,mad,len(depths),
            "VALID" if mad<=float(max_mad_m) else "GLOBAL_DEPTH_UNSTABLE",
            "V1_FIVE_POINT_MEDIAN",point_depths,positions)
    else:
        v1=AnchorDepthResult(False,valid_samples=len(depths),status="INSUFFICIENT_ANCHOR_DEPTH",
                             method="V1_FIVE_POINT_MEDIAN",point_depths_m=point_depths,
                             point_positions_px=positions)
    values=np.asarray(pooled,dtype=float)
    if len(values):
        low,high=np.percentile(values,[float(lower_percentile),float(upper_percentile)])
        trimmed=values[(values>=low)&(values<=high)]
        center=float(np.median(trimmed)); mad=float(np.median(np.abs(trimmed-center)))
        scale=max(1.4826*mad,1e-6)
        accepted=trimmed[np.abs(trimmed-center)<=float(outlier_mad_scale)*scale]
    else: accepted=np.empty(0); mad=None
    minimum_pixels=int(minimum_samples)*max(3,int(patch_size))
    if len(accepted)>=minimum_pixels:
        median=float(np.median(accepted)); final_mad=float(np.median(np.abs(accepted-median)))
        valid=final_mad<=float(max_mad_m)
        v2=AnchorDepthResult(valid,median,final_mad,len(accepted),
            "VALID" if valid else "GLOBAL_DEPTH_UNSTABLE","V2_POOLED_ROBUST",
            point_depths,positions)
    else:
        v2=AnchorDepthResult(False,valid_samples=len(accepted),status="INSUFFICIENT_ANCHOR_DEPTH",
                             method="V2_POOLED_ROBUST",point_depths_m=point_depths,
                             point_positions_px=positions)
    return v1,v2


class AnchorTemporalFilter:
    def __init__(self,enabled=True,alpha=.3,max_jump_m=.10,max_invalid_frames=5):
        self.enabled=bool(enabled); self.alpha=float(alpha); self.max_jump=float(max_jump_m)
        self.max_invalid=int(max_invalid_frames); self.value=None; self.invalid_count=0

    def update(self,anchor):
        if not anchor.valid or anchor.median_m is None:
            self.invalid_count+=1
            if self.invalid_count>=self.max_invalid: self.value=None
            return None,"GLOBAL_DEPTH_UNSTABLE"
        raw=float(anchor.median_m)
        if self.value is not None and abs(raw-self.value)>self.max_jump:
            self.invalid_count+=1
            if self.invalid_count>=self.max_invalid: self.value=None
            return None,"ANCHOR_DEPTH_JUMP"
        self.invalid_count=0
        self.value=raw if not self.enabled or self.value is None else self.alpha*raw+(1-self.alpha)*self.value
        return self.value,"VALID"

    def reset(self): self.value=None; self.invalid_count=0


def reconstruct_tip_camera(anchor_camera,rotation,tip_hand,anchor_hand):
    anchor=np.asarray(anchor_camera,dtype=float); rotation=np.asarray(rotation,dtype=float)
    delta=np.asarray(tip_hand,dtype=float)-np.asarray(anchor_hand,dtype=float)
    value=anchor+rotation@delta
    if anchor.shape!=(3,) or rotation.shape!=(3,3) or not np.all(np.isfinite(value)):
        raise ValueError("Cannot reconstruct finite TIP camera position")
    return value


def camera_ray(origin,direction):
    origin=np.asarray(origin,dtype=float); direction=_normalize(direction)
    if origin.shape!=(3,) or not np.all(np.isfinite(origin)) or direction is None:
        raise ValueError("Ray requires a finite origin and direction")
    return origin,direction


def world_relative_scale(world_landmarks,anchor_index=9):
    world=np.asarray(world_landmarks,dtype=float)
    if world.shape!=(21,3) or not np.all(np.isfinite(world)):
        raise ValueError("World landmarks must be finite 21x3")
    return {"tip_anchor_distance_m":float(np.linalg.norm(world[8]-world[int(anchor_index)])),
            "finger_length_m":float(sum(np.linalg.norm(world[b]-world[a])
                for a,b in ((5,6),(6,7),(7,8))))}


def scale_stability(values):
    array=np.asarray(values,dtype=float); array=array[np.isfinite(array)]
    if not len(array): return {"mean":None,"std":None,"cv":None}
    mean=float(np.mean(array)); std=float(np.std(array))
    return {"mean":mean,"std":std,"cv":std/mean if abs(mean)>1e-12 else None}


class DirectionEMA:
    def __init__(self,enabled=True,alpha=.35,max_angle_jump_deg=20):
        self.enabled=bool(enabled); self.alpha=float(alpha)
        self.max_jump=float(max_angle_jump_deg); self.previous={}

    def update(self,name,raw):
        raw=_normalize(raw)
        if raw is None: return None,None,False
        previous=self.previous.get(name)
        if not self.enabled or previous is None:
            self.previous[name]=raw; return raw,None,False
        jump=float(np.degrees(np.arccos(np.clip(np.dot(previous,raw),-1,1))))
        exceeded=jump>self.max_jump
        # Reset on a large jump so smoothing cannot conceal an actual reversal.
        smoothed=raw if exceeded else _normalize(self.alpha*raw+(1-self.alpha)*previous)
        self.previous[name]=smoothed
        return smoothed,jump,exceeded

    def reset(self): self.previous.clear()


class ExperimentalPointingCore:
    """Phase 2A A/B/C rays. Candidate C never consumes TIP/PIP scene-depth delta."""
    TIP=8; PIP=6

    def __init__(self,config):
        self.config=config
        temporal=config["temporal"]
        self.temporal=DirectionEMA(temporal["enabled"],temporal["ema_alpha"],
                                   temporal["max_angle_jump_deg"])
        anchor_temporal=config.get("anchor_temporal",{})
        self.anchor_temporal=AnchorTemporalFilter(anchor_temporal.get("enabled",True),
            anchor_temporal.get("ema_alpha",.3),anchor_temporal.get("max_jump_m",.10),
            anchor_temporal.get("max_invalid_frames",5))

    def reset(self): self.temporal.reset(); self.anchor_temporal.reset()

    def _quality(self,direction,base_quality="GOOD",marginal=False,jump=False):
        direction=_normalize(direction)
        if direction is None: return "INVALID"
        if direction[2]>0: return "AWAY"
        if marginal or jump: return "MARGINAL"
        return base_quality

    def _candidate(self,name,origin,raw,base_quality="GOOD",marginal=False):
        if origin is None or raw is None:
            return RayCandidate(name,quality=base_quality if base_quality!="GOOD" else "INVALID")
        smoothed,jump,exceeded=self.temporal.update(name,raw)
        return RayCandidate(name,np.asarray(origin,dtype=float),_normalize(raw),smoothed,
                            _angle_to_camera(smoothed),
                            self._quality(smoothed,base_quality,marginal,exceeded),jump)

    def process(self,world_landmarks,image_landmarks_px,depth_map,intrinsics,corrector=None):
        result=PointingCoreResult(intrinsics_mode=intrinsics.mode)
        if world_landmarks is None or image_landmarks_px is None:
            result.candidates={name:RayCandidate(name) for name in
                               ("A_BASELINE","B_RELATIVE","C_AXIS_FIT")}
            return result
        world=np.asarray(world_landmarks,dtype=float); pixels=np.asarray(image_landmarks_px,dtype=float)
        pnp_cfg=self.config["pnp"]
        mapping=estimate_hand_to_camera_rotation(world,pixels,intrinsics,
                                                  pnp_cfg["max_reprojection_rmse_px"])
        result.pnp_status=mapping.status; result.pnp_rmse_px=mapping.reprojection_rmse_px
        anchor_cfg=self.config["anchor"]
        v1,v2=palm_anchor_candidates(depth_map,pixels,anchor_cfg["sample_landmark_indices"],
            anchor_cfg["patch_size"],anchor_cfg["minimum_valid_samples"],
            anchor_cfg["max_depth_mad_m"],corrector,
            anchor_cfg.get("lower_percentile",10),anchor_cfg.get("upper_percentile",90),
            anchor_cfg.get("outlier_mad_scale",3.5))
        result.anchor_v1=v1; result.anchor_v2=v2
        anchor=v2 if anchor_cfg.get("mode","V2_POOLED_ROBUST")=="V2_POOLED_ROBUST" else v1
        result.anchor=anchor; result.raw_anchor_depth_m=anchor.median_m
        filtered_depth,temporal_status=self.anchor_temporal.update(anchor)
        result.filtered_anchor_depth_m=filtered_depth; result.anchor_temporal_status=temporal_status
        anchor_index=int(anchor_cfg["landmark_index"])
        if anchor.valid and anchor.median_m is not None:
            result.raw_anchor_camera=backproject_distorted_pixel(
                *pixels[anchor_index,:2],anchor.median_m,intrinsics)
        if filtered_depth is not None:
            result.filtered_anchor_camera=backproject_distorted_pixel(
                *pixels[anchor_index,:2],filtered_depth,intrinsics)
            result.anchor_camera=result.filtered_anchor_camera
        scale=world_relative_scale(world,anchor_index)
        result.world_tip_anchor_distance_m=scale["tip_anchor_distance_m"]
        result.world_finger_length_m=scale["finger_length_m"]

        # Candidate A is permanently preserved as the scene-depth baseline.
        pip_depth=sample_point(depth_map,pixels[self.PIP,:2],anchor_cfg["patch_size"])
        tip_depth=sample_point(depth_map,pixels[self.TIP,:2],anchor_cfg["patch_size"])
        pip_depth=corrector(pip_depth) if corrector and pip_depth is not None else pip_depth
        tip_depth=corrector(tip_depth) if corrector and tip_depth is not None else tip_depth
        scene_pip=(backproject_distorted_pixel(*pixels[self.PIP,:2],pip_depth,intrinsics)
                   if pip_depth is not None else None)
        scene_tip=(backproject_distorted_pixel(*pixels[self.TIP,:2],tip_depth,intrinsics)
                   if tip_depth is not None else None)
        baseline=_normalize(scene_tip-scene_pip) if scene_tip is not None and scene_pip is not None else None
        result.candidates["A_BASELINE"]=self._candidate("A_BASELINE",scene_tip,baseline)

        if not mapping.valid:
            for name in ("B_RELATIVE","C_AXIS_FIT"):
                result.candidates[name]=RayCandidate(name,quality="PNP_UNRELIABLE")
            return result
        if not anchor.valid or result.anchor_camera is None or temporal_status!="VALID":
            for name in ("B_RELATIVE","C_AXIS_FIT"):
                result.candidates[name]=RayCandidate(name,quality="GLOBAL_DEPTH_UNSTABLE")
            return result
        try:
            result.tip_camera=reconstruct_tip_camera(result.anchor_camera,mapping.rotation,
                                                      world[self.TIP],world[anchor_index])
        except ValueError:
            for name in ("B_RELATIVE","C_AXIS_FIT"):
                result.candidates[name]=RayCandidate(name,quality="INVALID")
            return result
        pnp_marginal=mapping.reprojection_rmse_px>float(pnp_cfg["max_reprojection_rmse_px"])*.75
        anchor_marginal=anchor.mad_m>float(anchor_cfg["max_depth_mad_m"])*.75
        b_raw=_normalize(mapping.rotation@(world[self.TIP]-world[self.PIP]))
        result.candidates["B_RELATIVE"]=self._candidate(
            "B_RELATIVE",result.tip_camera,b_raw,marginal=pnp_marginal or anchor_marginal)
        axis=fit_finger_axis_with_quality(world)
        result.axis_residual_m=axis.residual_m; result.axis_linearity=axis.linearity
        axis_cfg=self.config["axis"]
        axis_bad=(not axis.valid or axis.residual_m>float(axis_cfg["max_residual_m"])
                  or axis.linearity<float(axis_cfg["minimum_linearity"]))
        if axis_bad:
            result.candidates["C_AXIS_FIT"]=RayCandidate("C_AXIS_FIT",quality="HAND_AXIS_UNSTABLE")
        else:
            c_raw=_normalize(mapping.rotation@axis.direction)
            axis_marginal=(axis.residual_m>float(axis_cfg["max_residual_m"])*.75 or
                           axis.linearity<float(axis_cfg["minimum_linearity"])+.05)
            result.candidates["C_AXIS_FIT"]=self._candidate(
                "C_AXIS_FIT",result.tip_camera,c_raw,
                marginal=pnp_marginal or anchor_marginal or axis_marginal)
        return result
