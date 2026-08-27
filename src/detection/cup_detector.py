from __future__ import annotations

import cv2
import numpy as np

from src.detection.cup_tracker import CupTracker
from src.types import CupDetection


class CupDetector:
    CUP_LABEL = 47

    def __init__(self, confidence=0.38, max_cups=6, tracker_distance_px=150,
                 track_ttl_frames=20, device="auto"):
        try:
            import torch
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Torch/torchvision detector is unavailable") from exc
        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.confidence, self.max_cups = float(confidence), int(max_cups)
        self.tracker = CupTracker(tracker_distance_px, track_ttl_frames)

    def process(self, frame_bgr) -> list[CupDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            output = self.model([tensor])[0]
        cups = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if float(score) < self.confidence:
                break
            if int(label) != self.CUP_LABEL:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.detach().cpu().tolist()]
            cups.append(CupDetection(0, (x1, y1, x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2), float(score)))
            if len(cups) >= self.max_cups:
                break
        return self.tracker.update(cups)

