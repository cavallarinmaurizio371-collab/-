import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import isolate_runtime, load_yaml
isolate_runtime()
from src.safety.path_guard import PROJECT_ROOT


def main():
    cfg=load_yaml(PROJECT_ROOT/"configs/default.yaml")
    hand_model=PROJECT_ROOT/"models/mediapipe/hand_landmarker.task"
    if not hand_model.exists():
        raise FileNotFoundError(
            f"Missing {hand_model}. Download it with scripts/download_models.ps1 first.")
    print("Hand Landmarker model ready:", hand_model)
    print("Downloading detector weights into project/models...")
    from src.detection.cup_detector import CupDetector
    CupDetector(**cfg["detection"])
    print("Downloading metric depth weights into project/models...")
    from src.depth.depth_estimator import DepthEstimator
    model=DepthEstimator(cfg["depth"]["model_id"],"metric",cfg["depth"]["fallback_depth_m"],
                         local_files_only=False)
    print("Models ready:",model.model_id)


if __name__=="__main__": main()
