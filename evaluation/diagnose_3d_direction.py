from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.direction_candidates import build_candidates
from evaluation.direction_candidates.hand_probe import MediaPipeHandProbe
from src.camera.calibration import load_intrinsics
from src.camera.camera import Camera
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_estimator import DepthEstimator
from src.depth.depth_sampler import sample_point
from src.geometry.backprojection import pixel_depth_to_camera_xyz
from src.runtime import load_yaml
from src.safety.path_guard import safe_mkdir,safe_open
from src.visualization.renderer import CONNECTIONS


POSES=("CENTER","LEFT_TO_CENTER","RIGHT_TO_CENTER")
FIELDS=("timestamp","pose","hand_detected","pointing","gesture_confidence","handedness",
        "handedness_score","scene_pip_z_m","scene_tip_z_m","relative_pip_z_m",
        "relative_tip_z_m","pnp_status","pnp_rmse_px","candidate","native_dx","native_dy",
        "native_dz","camera_dx","camera_dy","camera_dz","angle_to_camera_deg","status")


def _text(image,value,xy,color=(255,255,255),scale=.5,thickness=1):
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+3,cv2.LINE_AA)
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def _angle(direction):
    if direction is None:
        return None
    value=np.asarray(direction,dtype=float); value=value/np.linalg.norm(value)
    return float(np.degrees(np.arccos(np.clip(np.dot(value,[0,0,-1]),-1,1))))


def _fmt(value,digits=4):
    return "N/A" if value is None or not np.isfinite(value) else f"{float(value):+.{digits}f}"


def _candidate_rows(timestamp,pose,hand,scene_points,candidates,mapping):
    pip_world=hand.world_landmarks_m[6] if hand.world_landmarks_m is not None else None
    tip_world=hand.world_landmarks_m[8] if hand.world_landmarks_m is not None else None
    common={"timestamp":timestamp,"pose":pose,"hand_detected":hand.detected,
            "pointing":hand.is_pointing,"gesture_confidence":hand.gesture_confidence,
            "handedness":hand.handedness,"handedness_score":hand.handedness_score,
            "scene_pip_z_m":scene_points.get("pip")[2] if scene_points.get("pip") is not None else None,
            "scene_tip_z_m":scene_points.get("tip")[2] if scene_points.get("tip") is not None else None,
            "relative_pip_z_m":pip_world[2] if pip_world is not None else None,
            "relative_tip_z_m":tip_world[2] if tip_world is not None else None,
            "pnp_status":mapping.status,"pnp_rmse_px":mapping.reprojection_rmse_px}
    rows=[]
    for name,candidate in candidates.items():
        native=candidate.native_direction; camera=candidate.camera_direction
        row={**common,"candidate":name,
             "native_dx":native[0] if native is not None else None,
             "native_dy":native[1] if native is not None else None,
             "native_dz":native[2] if native is not None else None,
             "camera_dx":camera[0] if camera is not None else None,
             "camera_dy":camera[1] if camera is not None else None,
             "camera_dz":camera[2] if camera is not None else None,
             "angle_to_camera_deg":_angle(camera),"status":candidate.camera_status}
        rows.append(row)
    return rows


def _aggregate(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row["pose"],row["candidate"])].append(row)
    result={}
    for (pose,candidate),items in groups.items():
        directions=[np.asarray([row["camera_dx"],row["camera_dy"],row["camera_dz"]],float)
                    for row in items if row["camera_dz"] is not None]
        directions=[value/np.linalg.norm(value) for value in directions if np.linalg.norm(value)>1e-9]
        toward=sum(row["status"]=="TOWARD" for row in items)
        if directions:
            matrix=np.stack(directions); mean=matrix.mean(axis=0); mean/=np.linalg.norm(mean)
            angles=np.degrees(np.arccos(np.clip(matrix@mean,-1,1)))
            jitter=float(np.std(angles)); negative=float(np.mean(matrix[:,2]<0))
            sign_consistency=max(negative,1-negative)
        else:
            jitter=sign_consistency=None
        result[f"{pose}/{candidate}"]={"frames":len(items),
            "mapped_direction_frames":len(directions),
            "toward_camera_rate":toward/len(items) if items else None,
            "direction_jitter_deg":jitter,"dz_sign_consistency":sign_consistency}
    return result


def _write_session(session_dir,rows,smoke):
    with safe_open(session_dir/"samples.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS); writer.writeheader()
        for row in rows: writer.writerow({key:row.get(key) for key in FIELDS})
    summary={"runtime_smoke":smoke,"captured_candidate_rows":len(rows),"captured_video_frames":len(rows)//3,
             "by_pose_and_candidate":_aggregate(rows)}
    with safe_open(session_dir/"summary.json","w",encoding="utf-8") as handle:
        json.dump(summary,handle,indent=2,ensure_ascii=False)
    lines=["# 3D Direction Candidate Minimal Check","",
           "> 这是 Phase 1 小样本诊断，不是最终准确率。","",
           f"Captured video frames: {summary['captured_video_frames']}","",
           "## Runtime smoke","","```json",json.dumps(smoke,indent=2,ensure_ascii=False),"```","",
           "## Captured pose statistics","","```json",
           json.dumps(summary["by_pose_and_candidate"],indent=2,ensure_ascii=False),"```"]
    with safe_open(session_dir/"summary.md","w",encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description="Phase 1 independent 3D direction candidate diagnostic")
    parser.add_argument("--depth",choices=("metric","approximate","auto"),default="metric")
    parser.add_argument("--capture-seconds",type=float,default=4.0)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--max-frames",type=int,default=0)
    args=parser.parse_args()
    config=load_yaml(PROJECT_ROOT/"configs/default.yaml")
    camera_cfg=config["camera"]; hand_cfg=config["hand"]; depth_cfg=config["depth"]
    camera=Camera(camera_cfg["index"],camera_cfg["width"],camera_cfg["height"],camera_cfg["fps"])
    hand=MediaPipeHandProbe(**hand_cfg)
    depth=DepthEstimator(depth_cfg["model_id"],args.depth,depth_cfg["fallback_depth_m"])
    calibration=DepthCalibration.load(PROJECT_ROOT/depth_cfg["calibration_file"])
    session=safe_mkdir(PROJECT_ROOT/"reports"/"phase1_sessions"/
                       datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"))
    pose_index=0; recording_until=0.0; rows=[]; frame_index=0; previous=time.perf_counter()
    depth_ema={}
    smoke={"runtime_frames":0,"hand_detected_frames":0,"world_landmark_frames":0,
           "pnp_valid_frames":0,"depth_mode":depth.mode,"intrinsics_mode":None}
    try:
        camera.open(); ok,frame=camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        intrinsics=load_intrinsics(PROJECT_ROOT/camera_cfg["intrinsics"],frame.shape[1],frame.shape[0])
        smoke["intrinsics_mode"]=intrinsics.mode
        while ok:
            now=time.perf_counter(); probe=hand.process(frame); depth_map=depth.process(frame)
            scene={"pip":None,"tip":None}
            if probe.detected:
                for name,index in (("pip",6),("tip",8)):
                    sampled=sample_point(depth_map,probe.pixel_landmarks[index,:2],depth_cfg["patch_size"])
                    value=calibration.correct(sampled) if sampled is not None else None
                    if value is not None:
                        old=depth_ema.get(name,value); alpha=float(depth_cfg["ema_alpha"])
                        value=alpha*value+(1-alpha)*old; depth_ema[name]=value
                        scene[name]=pixel_depth_to_camera_xyz(*probe.pixel_landmarks[index,:2],value,intrinsics)
            candidates,mapping=build_candidates(scene["pip"],scene["tip"],
                probe.world_landmarks_m if probe.detected else None,
                probe.pixel_landmarks if probe.detected else None,intrinsics)
            smoke["runtime_frames"]+=1
            smoke["hand_detected_frames"]+=int(probe.detected)
            smoke["world_landmark_frames"]+=int(probe.world_landmarks_m is not None)
            smoke["pnp_valid_frames"]+=int(mapping.valid)
            timestamp=datetime.now().isoformat(); current_rows=_candidate_rows(
                timestamp,POSES[pose_index],probe,scene,candidates,mapping)
            recording=now<recording_until
            if recording: rows.extend(current_rows)
            display=frame.copy()
            if probe.detected:
                points=np.rint(probe.pixel_landmarks[:,:2]).astype(int)
                for start,end in CONNECTIONS: cv2.line(display,tuple(points[start]),tuple(points[end]),(80,230,80),2)
                for point in points: cv2.circle(display,tuple(point),3,(0,255,80),-1)
            cv2.rectangle(display,(0,0),(display.shape[1],250),(20,20,20),-1)
            fps=1/max(now-previous,1e-6); previous=now
            _text(display,"PHASE 1 - 3D DIRECTION CANDIDATES",(12,28),(0,255,255),.72,2)
            _text(display,f"Pose: {POSES[pose_index]} | {'RECORDING' if recording else 'IDLE'} | 1/2/3 pose | SPACE record | Q quit",(12,54),(220,220,220),.44)
            _text(display,f"Hand: {probe.handedness or 'N/A'} pointing={probe.is_pointing} conf={probe.gesture_confidence:.2f} | FPS {fps:.1f}",(12,80))
            scene_pip=scene["pip"]; scene_tip=scene["tip"]
            _text(display,f"Scene depth PIP Z={_fmt(scene_pip[2] if scene_pip is not None else None)}m  TIP Z={_fmt(scene_tip[2] if scene_tip is not None else None)}m",(12,106))
            world=probe.world_landmarks_m
            _text(display,f"Hand-relative PIP Z={_fmt(world[6,2] if world is not None else None)}m  TIP Z={_fmt(world[8,2] if world is not None else None)}m",(12,132))
            for line,(name,candidate) in enumerate(candidates.items()):
                dz=candidate.camera_direction[2] if candidate.camera_direction is not None else None
                _text(display,f"{name}: dz={_fmt(dz)} angle={_fmt(_angle(candidate.camera_direction),1)} deg  {candidate.camera_status}",(12,160+line*25),(255,210,80))
            _text(display,f"Relative->Camera: {mapping.status}  reprojection RMSE={_fmt(mapping.reprojection_rmse_px,1)} px",(650,106),(0,200,255),.46)
            frame_index+=1
            if not args.headless:
                cv2.imshow("Phase 1 3D Direction Diagnostic",display)
                key=cv2.waitKey(1)&0xff
                if key in (27,ord('q'),ord('Q')): break
                if key in (ord('1'),ord('2'),ord('3')) and not recording:
                    pose_index=key-ord('1')
                elif key==32 and not recording:
                    recording_until=time.perf_counter()+float(args.capture_seconds)
            if args.max_frames and frame_index>=args.max_frames: break
            ok,frame=camera.read()
    finally:
        hand.close(); camera.release(); cv2.destroyAllWindows(); _write_session(session,rows,smoke)
        print("Phase 1 diagnostic output:",session)


if __name__=="__main__": main()
