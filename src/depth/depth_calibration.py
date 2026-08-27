from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.safety.path_guard import safe_open


class DepthCalibration:
    def __init__(self, a=1.0, b=0.0, calibrated=False, samples=None):
        self.a, self.b, self.calibrated = float(a), float(b), bool(calibrated)
        self.samples = samples or []

    @classmethod
    def load(cls, path: str | Path):
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return cls(**json.load(handle))
        except (OSError, ValueError, TypeError):
            return cls()

    def correct(self, raw_depth):
        return self.a * raw_depth + self.b

    def fit(self, predicted: list[float], actual: list[float]):
        if len(predicted) < 2 or len(predicted) != len(actual):
            raise ValueError("At least two paired depth samples are required")
        self.a, self.b = [float(v) for v in np.polyfit(predicted, actual, 1)]
        self.calibrated = True
        self.samples = [{"predicted": p, "actual": a} for p, a in zip(predicted, actual)]

    def save(self, path: str | Path):
        with safe_open(path, "w", encoding="utf-8") as handle:
            json.dump({"a": self.a, "b": self.b, "calibrated": self.calibrated,
                       "samples": self.samples}, handle, indent=2)

