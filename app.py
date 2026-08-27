from __future__ import annotations

import argparse
import time

from src.runtime import isolate_runtime
isolate_runtime()

import cv2

from src.camera.camera import Camera
from src.fusion.pipeline import VisionPipeline
from src.recording import SessionRecorder
from src.runtime import load_yaml
from src.safety.path_guard import PROJECT_ROOT, safe_mkdir
from src.visualization.renderer import render


def main():
    parser = argparse.ArgumentParser(description="Hand pointing + cup detection + monocular 3D demo")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--depth", choices=["auto","metric","approximate"], default=None)
    parser.add_argument("--headless", action="store_true", help="Run without a display window")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = unlimited)")
    args = parser.parse_args()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    if args.camera is not None: cfg["camera"]["index"] = args.camera
    if args.depth is not None: cfg["depth"]["backend"] = args.depth
    for folder in ("cache","models","outputs","logs","third_party"):
        safe_mkdir(PROJECT_ROOT / folder)
    ccfg = cfg["camera"]
    camera = Camera(ccfg["index"], ccfg["width"], ccfg["height"], ccfg["fps"])
    pipeline = recorder = None
    samples, start = [], time.perf_counter()
    try:
        camera.open()
        ok, frame = camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        h,w = frame.shape[:2]
        pipeline = VisionPipeline(cfg, PROJECT_ROOT, (w,h))
        print("Initialization notes:", pipeline.init_errors or "all modules loaded")
        recorder = SessionRecorder(args.record, (w,h), min(ccfg["fps"], 20))
        previous = time.perf_counter()
        frame_count = 0
        while ok:
            result = pipeline.process(frame)
            now = time.perf_counter(); fps = 1/max(now-previous, 1e-6); previous = now
            samples.append(fps)
            display = render(frame.copy(), result, fps, cfg["visualization"]["ray_length_px"])
            recorder.write(display, result, fps)
            frame_count += 1
            if not args.headless:
                cv2.imshow("Gesture Pointing 3D Demo", display)
                if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')): break
            if args.max_frames and frame_count >= args.max_frames: break
            ok, frame = camera.read()
    finally:
        if pipeline: pipeline.close()
        camera.release(); cv2.destroyAllWindows()
        average_fps = sum(samples)/len(samples) if samples else 0
        if recorder: recorder.close(average_fps)
        if samples: print(f"Average processing FPS: {average_fps:.2f} ({len(samples)} frames)")


if __name__ == "__main__":
    main()
