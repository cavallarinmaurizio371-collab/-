from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float]
    image_width: int
    image_height: int
    valid_calibration: bool = False
    mode: str = "APPROXIMATE_INTRINSICS"

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.image_width, height / self.image_height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                                self.distortion, width, height, self.valid_calibration, self.mode)


def approximate_intrinsics(width: int, height: int, horizontal_fov_deg: float = 60.0):
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return CameraIntrinsics(fx, fx, width / 2.0, height / 2.0, [0.0] * 5,
                            width, height, False, "APPROXIMATE_INTRINSICS")


def load_intrinsics(path: str | Path, width: int, height: int) -> CameraIntrinsics:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            intr = CameraIntrinsics(**json.load(handle))
        return intr.scaled(width, height)
    except (OSError, ValueError, TypeError):
        return approximate_intrinsics(width, height)

