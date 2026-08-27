import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import isolate_runtime, load_yaml
isolate_runtime()
import cv2
from src.camera.camera import Camera
from src.detection.cup_detector import CupDetector
from src.safety.path_guard import PROJECT_ROOT


def main():
    cfg = load_yaml(PROJECT_ROOT / "configs/default.yaml")
    detector = CupDetector(**cfg["detection"])
    with Camera() as camera:
        while True:
            ok, frame = camera.read()
            if not ok: break
            cups = detector.process(frame)
            for cup in cups:
                x1,y1,x2,y2 = cup.bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(frame,f"Cup {cup.id} {cup.confidence:.2f}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.putText(frame,f"Cups: {len(cups)}",(12,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
            cv2.imshow("Cup Detector Test",frame)
            if cv2.waitKey(1)&0xFF in (27,ord('q'),ord('Q')): break
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
