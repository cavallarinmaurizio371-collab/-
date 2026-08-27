from __future__ import annotations

import cv2
import numpy as np


class DepthEstimator:
    """Depth Anything V2 Metric, with an explicitly-labelled demo fallback."""
    def __init__(self, model_id, backend="auto", fallback_depth_m=0.8, device="auto",
                 local_files_only=True):
        self.model_id, self.fallback_depth = model_id, float(fallback_depth_m)
        self.processor = self.model = None
        self.mode = "APPROXIMATE_CONSTANT"
        self.error = None
        if backend == "approximate":
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.torch = torch
            self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
            self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id, local_files_only=local_files_only).to(self.device).eval()
            self.mode = "METRIC_RAW"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if backend == "metric":
                raise RuntimeError(f"Metric depth model unavailable: {self.error}") from exc

    def process(self, frame_bgr) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.model is None:
            return np.full((h, w), self.fallback_depth, dtype=np.float32)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze()
        depth = prediction.detach().float().cpu().numpy()
        return np.maximum(depth, 0).astype(np.float32)
