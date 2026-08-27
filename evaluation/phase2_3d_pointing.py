from __future__ import annotations

import argparse,csv,json,sys,time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.direction_candidates.hand_probe import MediaPipeHandProbe
from src.camera.camera import Camera
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_estimator import DepthEstimator
from src.experimental_3d_pointing.core import ExperimentalPointingCore
from src.experimental_3d_pointing.intrinsics import load_phase2_intrinsics
from src.runtime import load_yaml
from src.safety.path_guard import safe_mkdir,safe_open
from src.visualization.renderer import CONNECTIONS

POSES=("CENTER","LEFT_TO_CENTER","RIGHT_TO_CENTER")
CANDIDATES=("A_BASELINE","B_RELATIVE","C_AXIS_FIT")
COLORS={"A_BASELINE":(0,165,255),"B_RELATIVE":(255,180,30),"C_AXIS_FIT":(30,30,255)}


def _text(image,value,xy,color=(255,255,255),scale=.46,thickness=1):
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+3,cv2.LINE_AA)
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def _fmt(value,digits=3):
    return "N/A" if value is None or not np.isfinite(value) else f"{float(value):+.{digits}f}"


def _encoded(value):
    if isinstance(value,np.ndarray): return json.dumps(value.tolist())
    return value


def _project(point,intrinsics):
    if point is None or point[2]<=1e-6: return None
    return (int(round(intrinsics.fx*point[0]/point[2]+intrinsics.cx)),
            int(round(intrinsics.fy*point[1]/point[2]+intrinsics.cy)))


def _draw_rays(image,result,intrinsics):
    for name in CANDIDATES:
        candidate=result.candidates.get(name)
        if not candidate or candidate.origin_camera is None or candidate.smoothed_direction_camera is None: continue
        start=_project(candidate.origin_camera,intrinsics)
        end=_project(candidate.origin_camera+.18*candidate.smoothed_direction_camera,intrinsics)
        if start and end:
            cv2.arrowedLine(image,start,end,COLORS[name],3,cv2.LINE_AA,tipLength=.08)


def _row(timestamp,pose,trial_id,hand,result,depth_mode):
    row={"timestamp":timestamp,"pose":pose,"trial_id":trial_id,
         "hand_detected":hand.detected,"pointing":hand.is_pointing,
         "gesture_confidence":hand.gesture_confidence,"depth_mode":depth_mode,
         "intrinsics_mode":result.intrinsics_mode,"pnp_status":result.pnp_status,
         "pnp_rmse_px":result.pnp_rmse_px,"anchor_depth_median":result.anchor.median_m,
         "anchor_depth_mad":result.anchor.mad_m,"valid_anchor_samples":result.anchor.valid_samples,
         "anchor_status":result.anchor.status,"anchor_camera":_encoded(result.anchor_camera),
         "anchor_method":result.anchor.method,
         "anchor_v1_valid":result.anchor_v1.valid,"anchor_v1_median":result.anchor_v1.median_m,
         "anchor_v1_mad":result.anchor_v1.mad_m,"anchor_v2_valid":result.anchor_v2.valid,
         "anchor_v2_median":result.anchor_v2.median_m,"anchor_v2_mad":result.anchor_v2.mad_m,
         "palm_point_depths":json.dumps(result.anchor_v1.point_depths_m),
         "palm_point_positions":json.dumps(result.anchor_v1.point_positions_px),
         "raw_anchor_depth":result.raw_anchor_depth_m,"filtered_anchor_depth":result.filtered_anchor_depth_m,
         "anchor_temporal_status":result.anchor_temporal_status,
         "raw_anchor_camera":_encoded(result.raw_anchor_camera),
         "filtered_anchor_camera":_encoded(result.filtered_anchor_camera),
         "tip_camera":_encoded(result.tip_camera),"axis_residual_m":result.axis_residual_m,
         "axis_linearity":result.axis_linearity,
         "world_tip_anchor_distance_m":result.world_tip_anchor_distance_m,
         "world_finger_length_m":result.world_finger_length_m}
    for name in CANDIDATES:
        candidate=result.candidates.get(name)
        prefix=name.lower()
        row[f"{prefix}_origin"]=_encoded(candidate.origin_camera if candidate else None)
        row[f"{prefix}_raw_direction"]=_encoded(candidate.raw_direction_camera if candidate else None)
        row[f"{prefix}_smoothed_direction"]=_encoded(candidate.smoothed_direction_camera if candidate else None)
        row[f"{prefix}_dz"]=(candidate.smoothed_direction_camera[2] if candidate and
                              candidate.smoothed_direction_camera is not None else None)
        row[f"{prefix}_angle_deg"]=candidate.angle_to_camera_deg if candidate else None
        row[f"{prefix}_quality"]=candidate.quality if candidate else "INVALID"
        row[f"{prefix}_temporal_jump_deg"]=candidate.temporal_angle_jump_deg if candidate else None
    return row


def _directions(items,name):
    key=f"{name.lower()}_smoothed_direction"; result=[]
    for item in items:
        value=item.get(key)
        if value:
            vector=np.asarray(json.loads(value),float); vector/=np.linalg.norm(vector); result.append(vector)
    return result


def _aggregate(rows):
    summary={}
    for pose in POSES:
        items=[row for row in rows if row["pose"]==pose]
        if not items: continue
        pnp=[float(row["pnp_rmse_px"]) for row in items if row["pnp_rmse_px"] is not None]
        origins=[np.asarray(json.loads(row["tip_camera"]),float) for row in items if row["tip_camera"]]
        within_trial=[]
        for trial_id in sorted(set(row["trial_id"] for row in items)):
            trial_origins=[np.asarray(json.loads(row["tip_camera"]),float) for row in items
                           if row["trial_id"]==trial_id and row["tip_camera"]]
            if trial_origins:
                matrix=np.stack(trial_origins); centered=matrix-matrix.mean(axis=0)
                within_trial.append(float(np.sqrt(np.mean(np.sum(centered*centered,axis=1)))))
        anchor_mad=[float(row["anchor_depth_mad"]) for row in items if row["anchor_depth_mad"] is not None]
        scale_tip=[float(row["world_tip_anchor_distance_m"]) for row in items if row.get("world_tip_anchor_distance_m")]
        scale_finger=[float(row["world_finger_length_m"]) for row in items if row.get("world_finger_length_m")]
        pose_result={"frames":len(items),"trials":len(set(row["trial_id"] for row in items)),
            "pnp_valid_rate":sum(row["pnp_status"].startswith("PNP_") and row["pnp_status"] not in
                ("PNP_UNRELIABLE","PNP_HIGH_REPROJECTION_ERROR","PNP_INVALID_INPUT") for row in items)/len(items),
            "pnp_rmse_px_median":float(np.median(pnp)) if pnp else None,
            "anchor_depth_mad_median":float(np.median(anchor_mad)) if anchor_mad else None,
            "ray_origin_stability_median_within_trial":float(np.median(within_trial)) if within_trial else None,
            "ray_origin_pooled_spread_m":float(np.sqrt(np.mean(np.sum((np.stack(origins)-np.mean(origins,axis=0))**2,axis=1)))) if origins else None,
            "ray_origin_valid_rate":len(origins)/len(items),
            "anchor_v1_valid_rate":sum(str(row.get("anchor_v1_valid"))=="True" for row in items)/len(items),
            "anchor_v2_valid_rate":sum(str(row.get("anchor_v2_valid"))=="True" for row in items)/len(items),
            "world_tip_anchor_scale":{"mean_m":float(np.mean(scale_tip)) if scale_tip else None,
                "std_m":float(np.std(scale_tip)) if scale_tip else None,
                "cv":float(np.std(scale_tip)/np.mean(scale_tip)) if scale_tip and np.mean(scale_tip) else None},
            "world_finger_length_scale":{"mean_m":float(np.mean(scale_finger)) if scale_finger else None,
                "std_m":float(np.std(scale_finger)) if scale_finger else None,
                "cv":float(np.std(scale_finger)/np.mean(scale_finger)) if scale_finger and np.mean(scale_finger) else None},
            "candidates":{}}
        for name in CANDIDATES:
            directions=_directions(items,name)
            if directions:
                matrix=np.stack(directions); mean=matrix.mean(axis=0); mean/=np.linalg.norm(mean)
                angles=np.degrees(np.arccos(np.clip(matrix@mean,-1,1)))
                toward=float(np.mean(matrix[:,2]<0))
                pose_result["candidates"][name]={"toward_camera_rate":toward,
                    "dz_sign_consistency":max(toward,1-toward),
                    "direction_jitter_deg":float(np.std(angles)),
                    "valid_direction_frames":len(directions),
                    "complete_ray_valid_rate":len(directions)/len(items)}
            else: pose_result["candidates"][name]={"toward_camera_rate":None,
                "dz_sign_consistency":None,"direction_jitter_deg":None,"valid_direction_frames":0}
        summary[pose]=pose_result
    return summary


def _write_outputs(session,rows,intrinsics_mode):
    fields=list(rows[0]) if rows else ["timestamp","pose","trial_id"]
    with safe_open(session/"samples.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary={"intrinsics_mode":intrinsics_mode,"captured_frames":len(rows),
             "by_pose":_aggregate(rows),"calibration_status":
             "COMPLETE" if intrinsics_mode=="CALIBRATED_INTRINSICS" else "PENDING_REAL_CAMERA_CALIBRATION"}
    with safe_open(session/"summary.json","w",encoding="utf-8") as handle:
        json.dump(summary,handle,indent=2,ensure_ascii=False)
    architecture=("Direction: MediaPipe world MCP/PIP/DIP/TIP -> PCA/TLS -> PnP rotation. "
                  "Distance: robust palm depth median -> anchor Camera XYZ. "
                  "TIP origin: anchor + R*(worldTIP-worldAnchor).")
    lines=["# Phase 2A 3D Pointing Report","",f"> {architecture}","",
           f"- Camera intrinsics: {intrinsics_mode}",f"- Calibration: {summary['calibration_status']}",
           f"- Captured frames: {len(rows)}","","## A/B/C Results","","```json",
           json.dumps(summary["by_pose"],indent=2,ensure_ascii=False),"```","",
           "## Remaining issues","",
           "真实内参未完成时，PnP、Camera-space origin 和毫米位置仍属于实验结果。"
           "本阶段不计算九区域 Accuracy，也不接入原 Demo。"]
    for path in (session/"summary.md",PROJECT_ROOT/"reports"/"3d_pointing_phase2a5.md"):
        with safe_open(path,"w",encoding="utf-8") as handle: handle.write("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description="Independent Phase 2A Camera-space pointing ray")
    parser.add_argument("--depth",choices=("metric","approximate","auto"),default="metric")
    parser.add_argument("--direction-mode",choices=CANDIDATES,default=None)
    parser.add_argument("--headless",action="store_true"); parser.add_argument("--max-frames",type=int,default=0)
    args=parser.parse_args()
    project=load_yaml(PROJECT_ROOT/"configs/default.yaml")
    config=load_yaml(PROJECT_ROOT/"configs/phase2_3d_pointing.yaml")
    camera_cfg=project["camera"]; hand_cfg=project["hand"]; depth_cfg=project["depth"]
    selected=args.direction_mode or config["runtime"]["direction_mode"]
    camera=Camera(camera_cfg["index"],camera_cfg["width"],camera_cfg["height"],camera_cfg["fps"])
    hand=MediaPipeHandProbe(**hand_cfg); depth=DepthEstimator(depth_cfg["model_id"],args.depth,depth_cfg["fallback_depth_m"])
    calibration=DepthCalibration.load(PROJECT_ROOT/depth_cfg["calibration_file"])
    core=ExperimentalPointingCore(config); rows=[]; pose_index=0; recording_until=0.; trial_counts=defaultdict(int)
    session=safe_mkdir(PROJECT_ROOT/config["runtime"]["output_root"]/
                       datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"))
    frame_count=0; previous=time.perf_counter(); intrinsics=None
    try:
        camera.open(); ok,frame=camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        intrinsics=load_phase2_intrinsics(PROJECT_ROOT,config,frame.shape[1],frame.shape[0])
        while ok:
            now=time.perf_counter(); probe=hand.process(frame); depth_map=depth.process(frame)
            result=core.process(probe.world_landmarks_m,probe.pixel_landmarks,depth_map,intrinsics,
                                calibration.correct if calibration.calibrated else None)
            recording=now<recording_until; pose=POSES[pose_index]
            if recording:
                rows.append(_row(datetime.now().isoformat(),pose,f"{pose}_{trial_counts[pose]:02d}",
                                 probe,result,depth.mode))
            display=frame.copy()
            if probe.detected:
                points=np.rint(probe.pixel_landmarks[:,:2]).astype(int)
                for start,end in CONNECTIONS: cv2.line(display,tuple(points[start]),tuple(points[end]),(80,220,80),2)
                for point in points: cv2.circle(display,tuple(point),3,(0,255,80),-1)
            _draw_rays(display,result,intrinsics)
            cv2.rectangle(display,(0,0),(display.shape[1],340),(20,20,20),-1)
            fps=1/max(now-previous,1e-6); previous=now
            _text(display,"PHASE 2A - CAMERA SPACE 3D RAY",(12,28),(0,255,255),.72,2)
            _text(display,f"Pose {pose} | {'RECORDING' if recording else 'IDLE'} | trials {dict(trial_counts)} | 1/2/3 | SPACE | Q",(12,54))
            _text(display,f"Selected {selected} | FPS {fps:.1f} | Intrinsics {intrinsics.mode}",(12,78))
            for index,name in enumerate(CANDIDATES):
                candidate=result.candidates.get(name); direction=candidate.smoothed_direction_camera if candidate else None
                _text(display,f"{name}: dz={_fmt(direction[2] if direction is not None else None)} angle={_fmt(candidate.angle_to_camera_deg if candidate else None,1)} quality={candidate.quality if candidate else 'INVALID'}",
                      (12,108+index*25),COLORS[name])
            _text(display,f"PnP {result.pnp_status} RMSE={_fmt(result.pnp_rmse_px,1)}px",(12,190))
            _text(display,f"V1 Z={_fmt(result.anchor_v1.median_m)} MAD={_fmt(result.anchor_v1.mad_m)} {result.anchor_v1.status} | V2 Z={_fmt(result.anchor_v2.median_m)} MAD={_fmt(result.anchor_v2.mad_m)} {result.anchor_v2.status}",(12,216))
            _text(display,f"Anchor raw/filtered Z={_fmt(result.raw_anchor_depth_m)}/{_fmt(result.filtered_anchor_depth_m)}m temporal={result.anchor_temporal_status}",(12,242))
            _text(display,f"Anchor XYZ={_fmt(result.anchor_camera[0] if result.anchor_camera is not None else None)},{_fmt(result.anchor_camera[1] if result.anchor_camera is not None else None)},{_fmt(result.anchor_camera[2] if result.anchor_camera is not None else None)}",(12,266))
            _text(display,f"TIP XYZ={_fmt(result.tip_camera[0] if result.tip_camera is not None else None)},{_fmt(result.tip_camera[1] if result.tip_camera is not None else None)},{_fmt(result.tip_camera[2] if result.tip_camera is not None else None)}",(12,290))
            _text(display,f"C residual={_fmt(result.axis_residual_m)}m linearity={_fmt(result.axis_linearity)} scale={_fmt(result.world_tip_anchor_distance_m)}m",(12,314))
            frame_count+=1
            if not args.headless:
                cv2.imshow("Phase 2A 3D Pointing",display); key=cv2.waitKey(1)&0xff
                if key in (27,ord('q'),ord('Q')): break
                if key in (ord('1'),ord('2'),ord('3')) and not recording:
                    pose_index=key-ord('1'); core.reset()
                elif key==32 and not recording:
                    trial_counts[pose]+=1; core.reset()
                    recording_until=time.perf_counter()+float(config["runtime"]["capture_seconds"])
            if args.max_frames and frame_count>=args.max_frames: break
            ok,frame=camera.read()
    finally:
        hand.close(); camera.release(); cv2.destroyAllWindows()
        _write_outputs(session,rows,intrinsics.mode if intrinsics else "UNKNOWN")
        print("Phase 2A output:",session)


if __name__=="__main__": main()
