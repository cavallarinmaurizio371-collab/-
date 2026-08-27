import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import isolate_runtime, load_yaml
isolate_runtime()
import cv2
from src.camera.camera import Camera
from src.hand.hand_landmarker import HandLandmarker
from src.safety.path_guard import PROJECT_ROOT
from src.types import PipelineResult
from src.visualization.renderer import render


def main():
    cfg = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    landmarker = HandLandmarker(**cfg["hand"])
    with Camera() as camera:
        while True:
            ok, frame = camera.read()
            if not ok: break
            hand = landmarker.process(frame)
            result = PipelineResult(hand, [], None, None, None, "NOT_TESTED",
                                    diagnostics={"calibration":"NOT_TESTED","init_errors":{}})
            cv2.imshow("Hand Test", render(frame, result, 0, cfg["visualization"]["ray_length_px"]))
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')): break
    landmarker.close(); cv2.destroyAllWindows()


if __name__ == "__main__": main()
