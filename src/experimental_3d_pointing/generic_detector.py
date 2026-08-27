from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectDetection:
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center_2d: tuple[float, float]
    raw_depth_m: float | None = None
    depth_m: float | None = None
    center_camera: np.ndarray | None = None
    distance_m: float | None = None
    depth_valid: bool = False


@dataclass(frozen=True)
class ObjectClassConfiguration:
    enabled: bool
    all_classes: bool
    included_classes: tuple[str, ...]
    excluded_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]


@dataclass(frozen=True)
class RobustObjectDepth:
    valid: bool
    median_m: float | None = None
    mad_m: float | None = None
    valid_samples: int = 0


def robust_bbox_depth(depth_map, bbox, inner_ratio=.5, lower_percentile=10,
                      upper_percentile=90, outlier_mad_scale=3.5):
    if depth_map is None:
        return RobustObjectDepth(False)
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = max(1.0, (x2 - x1) * float(inner_ratio) / 2.0)
    half_height = max(1.0, (y2 - y1) * float(inner_ratio) / 2.0)
    height, width = depth_map.shape[:2]
    xa, xb = max(0, int(cx - half_width)), min(width, int(cx + half_width) + 1)
    ya, yb = max(0, int(cy - half_height)), min(height, int(cy + half_height) + 1)
    values = np.asarray(depth_map[ya:yb, xa:xb], dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values): return RobustObjectDepth(False)
    low, high = np.percentile(values, [float(lower_percentile), float(upper_percentile)])
    trimmed = values[(values >= low) & (values <= high)]
    if not len(trimmed): return RobustObjectDepth(False)
    center = float(np.median(trimmed)); mad = float(np.median(np.abs(trimmed - center)))
    scale = max(1.4826 * mad, 1e-6)
    accepted = trimmed[np.abs(trimmed - center) <= float(outlier_mad_scale) * scale]
    if not len(accepted): return RobustObjectDepth(False)
    median = float(np.median(accepted)); final_mad = float(np.median(np.abs(accepted - median)))
    return RobustObjectDepth(True, median, final_mad, int(len(accepted)))


def coco_supported_classes():
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    categories = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta["categories"]
    return tuple(name for name in categories if name not in ("__background__", "N/A"))


def resolve_object_class_configuration(supported_classes, enabled=True,
                                       target_classes=None, excluded_classes=None,
                                       use_all_supported_classes=False):
    supported = tuple(str(value) for value in supported_classes)
    lookup = {value.lower(): value for value in supported}
    excluded_input = tuple(str(value) for value in (excluded_classes or []))
    excluded = tuple(lookup.get(value.lower(), value) for value in excluded_input)
    if not enabled:
        return ObjectClassConfiguration(False, False, (), excluded, ())
    all_requested = bool(use_all_supported_classes) or (
        isinstance(target_classes, str) and target_classes.strip().upper() == "ALL")
    if all_requested:
        return ObjectClassConfiguration(True, True, supported, excluded, ())
    requested = tuple(str(value) for value in (target_classes or []))
    unknown = tuple(value for value in requested if value.lower() not in lookup)
    included = tuple(lookup[value.lower()] for value in requested if value.lower() in lookup)
    return ObjectClassConfiguration(bool(included), False, included, excluded, unknown)


class GenericObjectTracker:
    def __init__(self, max_distance_px=150, ttl_frames=20):
        self.max_distance = float(max_distance_px); self.ttl = int(ttl_frames)
        self.tracks = {}; self.next_id = 1

    def update(self, detections):
        unmatched = set(self.tracks)
        for detection in detections:
            center = np.asarray(detection.center_2d, dtype=float)
            candidates = []
            for track_id in unmatched:
                class_id, old_center, _ = self.tracks[track_id]
                if class_id == detection.class_id:
                    candidates.append((float(np.linalg.norm(center - old_center)), track_id))
            distance, match = min(candidates, default=(float("inf"), None))
            if match is not None and distance <= self.max_distance:
                detection.id = match; unmatched.remove(match)
            else:
                detection.id = self.next_id; self.next_id += 1
            self.tracks[detection.id] = (detection.class_id, center, 0)
        for track_id in list(unmatched):
            class_id, center, age = self.tracks[track_id]
            if age + 1 >= self.ttl: del self.tracks[track_id]
            else: self.tracks[track_id] = (class_id, center, age + 1)
        return detections


class GenericObjectDetector:
    """COCO SSDLite detector with no cup-only business filter."""

    def __init__(self, confidence_threshold=.38, max_objects=12, tracker_distance_px=150,
                 track_ttl_frames=20, target_classes="ALL", excluded_classes=None,
                 use_all_supported_classes=False, device="auto"):
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )
        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = tuple(weights.meta["categories"])
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.confidence = float(confidence_threshold); self.max_objects = int(max_objects)
        self.class_config = resolve_object_class_configuration(
            self.supported_classes, True, target_classes, excluded_classes,
            use_all_supported_classes)
        self.included = {value.lower() for value in self.class_config.included_classes}
        self.excluded = {value.lower() for value in self.class_config.excluded_classes}
        self.tracker = GenericObjectTracker(tracker_distance_px, track_ttl_frames)

    @property
    def supported_classes(self):
        return tuple(name for name in self.categories if name not in ("__background__", "N/A"))

    def class_allowed(self, class_id, class_name):
        name = str(class_name).lower(); identifier = str(int(class_id))
        if name in ("__background__", "n/a"):
            return False
        class_config = getattr(self, "class_config", None)
        if class_config is not None and not class_config.enabled:
            return False
        all_classes = class_config.all_classes if class_config is not None else not self.included
        if not all_classes and name not in self.included and identifier not in self.included:
            return False
        return name not in self.excluded and identifier not in self.excluded

    def decode_output(self, output):
        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            confidence = float(score.detach().cpu() if hasattr(score, "detach") else score)
            if confidence < self.confidence: break
            class_id = int(label.detach().cpu() if hasattr(label, "detach") else label)
            name = self.categories[class_id] if 0 <= class_id < len(self.categories) else f"class_{class_id}"
            if not self.class_allowed(class_id, name): continue
            values = box.detach().cpu().tolist() if hasattr(box, "detach") else box
            x1, y1, x2, y2 = [int(round(float(value))) for value in values]
            if x2 <= x1 or y2 <= y1: continue
            detections.append(ObjectDetection(0, class_id, name, confidence,
                (x1, y1, x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
            if len(detections) >= self.max_objects: break
        return detections

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode(): output = self.model([tensor])[0]
        return self.tracker.update(self.decode_output(output))
