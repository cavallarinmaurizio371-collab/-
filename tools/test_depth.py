import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import isolate_runtime, load_yaml
isolate_runtime()
import cv2, numpy as np
from src.camera.camera import Camera
from src.depth.depth_estimator import DepthEstimator
from src.safety.path_guard import PROJECT_ROOT


def main():
    cfg=load_yaml(PROJECT_ROOT/"configs/default.yaml"); d=cfg["depth"]
    estimator=DepthEstimator(d["model_id"],d["backend"],d["fallback_depth_m"])
    mouse=[0,0]
    def move(event,x,y,*_):
        if event==cv2.EVENT_MOUSEMOVE: mouse[:]=[x,y]
    cv2.namedWindow("Depth Test"); cv2.setMouseCallback("Depth Test",move)
    with Camera() as camera:
        while True:
            ok,frame=camera.read()
            if not ok: break
            t=time.perf_counter(); depth=estimator.process(frame); fps=1/max(time.perf_counter()-t,1e-6)
            lo,hi=np.percentile(depth,[2,98]); normalized=np.clip((depth-lo)/max(hi-lo,1e-6)*255,0,255).astype(np.uint8)
            color=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
            x,y=np.clip(mouse[0],0,depth.shape[1]-1),np.clip(mouse[1],0,depth.shape[0]-1)
            cv2.putText(color,f"{estimator.mode} cursor={depth[y,x]:.3f} FPS={fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            cv2.imshow("Depth Test",np.hstack((frame,color)))
            if cv2.waitKey(1)&0xFF in (27,ord('q'),ord('Q')): break
    cv2.destroyAllWindows()


if __name__=="__main__": main()
