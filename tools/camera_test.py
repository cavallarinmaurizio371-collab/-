import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import isolate_runtime
isolate_runtime()
import cv2
from src.camera.camera import Camera


def main():
    with Camera() as camera:
        while True:
            ok, frame = camera.read()
            if not ok: raise RuntimeError("Camera stopped returning frames")
            cv2.putText(frame, "Camera OK - Q/ESC to quit", (15,35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2)
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')): break
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
