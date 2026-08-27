"""Experimental Phase 2A pointing core; isolated from the production pipeline."""

from src.experimental_3d_pointing.core import (
    AnchorDepthResult, ExperimentalPointingCore, PointingCoreResult, RayCandidate,
    camera_ray, reconstruct_tip_camera, robust_anchor_depth,
    palm_anchor_candidates,scale_stability,world_relative_scale,
)

__all__=["AnchorDepthResult","ExperimentalPointingCore","PointingCoreResult",
         "RayCandidate","camera_ray","reconstruct_tip_camera","robust_anchor_depth",
         "palm_anchor_candidates","scale_stability","world_relative_scale"]
