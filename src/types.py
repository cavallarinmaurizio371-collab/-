from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HandState:
    detected: bool = False
    is_pointing: bool = False
    confidence: float = 0.0
    landmarks_2d: Optional[np.ndarray] = None
    index_mcp: Optional[np.ndarray] = None
    index_pip: Optional[np.ndarray] = None
    index_dip: Optional[np.ndarray] = None
    index_tip: Optional[np.ndarray] = None
    direction_2d: Optional[np.ndarray] = None


@dataclass
class CupDetection:
    id: int
    bbox: tuple[int, int, int, int]
    center_2d: tuple[float, float]
    confidence: float
    raw_depth: Optional[float] = None
    depth: Optional[float] = None
    center_3d: Optional[np.ndarray] = None
    track_age: int = 0


@dataclass
class PipelineResult:
    hand: HandState
    cups: list[CupDetection]
    selected_cup: Optional[int]
    selected_score: Optional[float]
    depth_map: Optional[np.ndarray]
    depth_mode: str
    tip_3d: Optional[np.ndarray] = None
    pip_3d: Optional[np.ndarray] = None
    ray_direction_3d: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)

