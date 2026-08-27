from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.coordinate_adapter import configured_targets,mirror_display_x
from evaluation.direction_candidates.hand_probe import MediaPipeHandProbe
from evaluation.phase2b_metrics import (CandidateHit,evaluate_camera_candidate,
    select_quality_fallback)
from evaluation.phase2b_recorder import Phase2BRecorder
from src.camera.camera import Camera
from src.depth.depth_calibration import DepthCalibration
from src.depth.depth_estimator import DepthEstimator
from src.experimental_3d_pointing.core import ExperimentalPointingCore
from src.experimental_3d_pointing.intrinsics import (load_phase2b_intrinsics,
    project_camera_points)
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path
from src.visualization.renderer import CONNECTIONS


CANDIDATE_MAP={"A":"A_BASELINE","B":"B_RELATIVE","C":"C_AXIS_FIT"}
COLORS={"A":(0,165,255),"B":(255,120,20),"C":(20,20,255)}


def _text(image,value,xy,color=(255,255,255),scale=.44,thickness=1):
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+3,cv2.LINE_AA)
    cv2.putText(image,str(value),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def _fmt(value,digits=1):
    return "N/A" if value is None or not np.isfinite(value) else f"{float(value):.{digits}f}"


def build_protocol_plan(config,protocol):
    formal=config["formal"]
    if protocol in ("smoke","diagnostic"):
        block=config[protocol]
        return [(target,float(formal["distance_cm"]),"CENTER",repeat+1)
                for target in block["targets"] for repeat in range(int(block["trials_per_target"]))]
    if protocol=="full":
        block=config["full"]
        return [(target,float(formal["distance_cm"]),"CENTER",repeat+1)
                for target in block["targets"] for repeat in range(int(block["trials_per_target"]))]
    if protocol=="hand-position":
        block=config["hand_position"]
        return [(block["target"],float(formal["distance_cm"]),position,repeat+1)
                for position in block["positions"] for repeat in range(int(block["trials_per_position"]))]
    block=config["distance_generalization"]
    return [(target,float(distance),"CENTER",repeat+1) for distance in block["distances_cm"]
            for target in block["targets"] for repeat in range(int(block["trials_per_target"]))]


def _draw_hand(display,probe,mirror):
    if not probe.detected: return
    width=display.shape[1]; points=np.rint(probe.pixel_landmarks[:,:2]).astype(int)
    if mirror: points[:,0]=width-1-points[:,0]
    for start,end in CONNECTIONS: cv2.line(display,tuple(points[start]),tuple(points[end]),(70,230,90),2)
    for index,point in enumerate(points):
        cv2.circle(display,tuple(point),5 if index in (5,6,7,8) else 3,
                   (0,255,255) if index in (5,6,7,8) else (0,255,80),-1)


def _draw_camera_rays(display,core,intrinsics,mirror):
    width=display.shape[1]
    for prefix,name in CANDIDATE_MAP.items():
        candidate=core.candidates.get(name)
        if not candidate or candidate.origin_camera is None or candidate.smoothed_direction_camera is None:
            continue
        distance=min(.18,max(.02,float(candidate.origin_camera[2])*.45))
        end=candidate.origin_camera+distance*candidate.smoothed_direction_camera
        if end[2]<=.02: end=candidate.origin_camera+.25*float(candidate.origin_camera[2])*candidate.smoothed_direction_camera
        try: pixels=project_camera_points([candidate.origin_camera,end],intrinsics)
        except ValueError: continue
        points=[]
        for x,y in pixels:
            points.append((int(round(mirror_display_x(x,width,mirror))),int(round(y))))
        cv2.arrowedLine(display,points[0],points[1],COLORS[prefix],3,cv2.LINE_AA,tipLength=.10)


def _target_panel(display,targets,gt,hits):
    height,width=display.shape[:2]; panel_w,panel_h=330,300
    x0,y0=width-panel_w-10,74
    cv2.rectangle(display,(x0,y0),(x0+panel_w,y0+panel_h),(25,25,25),-1)
    cv2.rectangle(display,(x0,y0),(x0+panel_w,y0+panel_h),(210,210,210),1)
    _text(display,"A4 Z=0 TARGET PLANE",(x0+10,y0+23),(0,255,255),.5,2)
    def panel(point):
        return (int(x0+panel_w/2+float(point[0])/210.0*panel_w*.88),
                int(y0+panel_h/2-float(point[1])/297.0*panel_h*.88))
    for name,point in targets.items():
        px,py=panel(point); _text(display,name.replace("RIGHT","R").replace("LEFT","L").replace("CENTER","C").replace("DOWN","D").replace("UP","U"),(px-10,py+4),(170,170,170),.32)
    if gt in targets: cv2.circle(display,panel(targets[gt]),7,(0,220,0),-1)
    for prefix,color in (("B",COLORS["B"]),("C",COLORS["C"])):
        hit=hits[prefix]
        if hit.in_target_range and hit.hit_eval_mm is not None:
            px,py=panel(hit.hit_eval_mm)
            cv2.line(display,(px-7,py-7),(px+7,py+7),color,3)
            cv2.line(display,(px-7,py+7),(px+7,py-7),color,3)
    _text(display,"GT green | B blue | C red",(x0+8,y0+panel_h-10),(210,210,210),.38)


def _frame_row(context,probe,core,hits,selected_name,selected_hit,depth_mode,mirror):
    row={"session_id":context["session_id"],"trial_id":context["trial_id"],
         "timestamp":datetime.now().isoformat(),"protocol":context["protocol"],
         "gt_target":context["gt_target"],"gt_x_mm":context["gt_xy"][0],
         "gt_y_mm":context["gt_xy"][1],"distance_cm":context["distance_cm"],
         "hand_position":context["hand_position"],"hand_detected":probe.detected,
         "pointing_state":probe.is_pointing,"gesture_confidence":probe.gesture_confidence,
         "intrinsics_mode":core.intrinsics_mode,"calibration_id":context["calibration_id"],
         "pnp_valid":core.pnp_status=="PNP_CALIBRATED_INTRINSICS",
         "pnp_rmse_px":core.pnp_rmse_px,"anchor_v2_valid":core.anchor_v2.valid,
         "raw_anchor_depth":core.raw_anchor_depth_m,
         "filtered_anchor_depth":core.filtered_anchor_depth_m,"anchor_mad":core.anchor_v2.mad_m,
         "raw_anchor_xyz":core.raw_anchor_camera,"filtered_anchor_xyz":core.filtered_anchor_camera,
         "tip_camera_xyz":core.tip_camera,"depth_mode":depth_mode,"mirror_display":mirror}
    for prefix,name in CANDIDATE_MAP.items():
        candidate=core.candidates.get(name); hit=hits[prefix]
        row.update({f"{prefix}_origin":candidate.origin_camera if candidate else None,
                    f"{prefix}_direction":candidate.smoothed_direction_camera if candidate else None,
                    f"{prefix}_quality":candidate.quality if candidate else "INVALID",
                    f"{prefix}_complete_ray_valid":hit.complete_ray_valid,
                    f"{prefix}_toward_camera":hit.toward_camera,
                    f"{prefix}_quality_valid":hit.direction_quality_valid,
                    f"{prefix}_hit_valid":hit.intersection_valid,
                    f"{prefix}_in_target_range":hit.in_target_range,
                    f"{prefix}_intersection_status":hit.status,
                    f"{prefix}_hit_xy_mm":hit.hit_eval_mm,
                    f"{prefix}_region":hit.pred_region,f"{prefix}_error_mm":hit.error_mm})
    row["selected_candidate"]=selected_name
    row["selected_hit_xy_mm"]=(selected_hit.hit_eval_mm if selected_hit else None)
    row["selected_error_mm"]=(selected_hit.error_mm if selected_hit else None)
    return row


def _write_screenshot(path,image):
    ok,encoded=cv2.imencode(".png",image)
    if not ok: raise RuntimeError("Could not encode screenshot")
    path.write_bytes(encoded.tobytes())


def main():
    parser=argparse.ArgumentParser(description="Phase 2B calibrated parallel Z=0 evaluation")
    parser.add_argument("--protocol",choices=("diagnostic","smoke","full","hand-position","distance"),default="smoke")
    parser.add_argument("--direction-mode",choices=("baseline","relative","axisfit","parallel"),default="parallel")
    parser.add_argument("--headless",action="store_true"); parser.add_argument("--max-frames",type=int,default=0)
    args=parser.parse_args()
    project=load_yaml(PROJECT_ROOT/"configs/default.yaml")
    config=load_yaml(PROJECT_ROOT/"configs/phase2b_z0.yaml")
    core_config=load_yaml(PROJECT_ROOT/config["core_config"])
    targets=configured_targets(load_yaml(PROJECT_ROOT/config["targets_file"]))
    formal=config["formal"]
    if set(targets)!={"LEFT_UP","UP","RIGHT_UP","LEFT","CENTER","RIGHT","LEFT_DOWN","DOWN","RIGHT_DOWN"}:
        raise RuntimeError("FORMAL_TARGET_CONFIG_INCOMPLETE")
    plan=build_protocol_plan(config,args.protocol); index=0
    camera_cfg=project["camera"].copy(); camera_cfg["width"]=config["intrinsics"]["required_width"]
    camera_cfg["height"]=config["intrinsics"]["required_height"]
    camera=Camera(camera_cfg["index"],camera_cfg["width"],camera_cfg["height"],camera_cfg["fps"])
    hand=MediaPipeHandProbe(**project["hand"])
    depth_cfg=project["depth"]
    depth=DepthEstimator(depth_cfg["model_id"],formal["depth_backend"],depth_cfg["fallback_depth_m"])
    depth_cal=DepthCalibration.load(PROJECT_ROOT/depth_cfg["calibration_file"])
    core=ExperimentalPointingCore(core_config)
    recorder=Phase2BRecorder(PROJECT_ROOT/config["output_root"],args.protocol,config["excluded_sessions"])
    mirror=bool(formal["mirror_display"]); state="IDLE"; deadline=0.0; rows=[]; frame_count=0
    intrinsics=None; previous=time.perf_counter(); message="Press SPACE to start current Trial"
    pending_screenshot=None; protocol_complete=False
    try:
        camera.open(); ok,frame=camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        if (frame.shape[1],frame.shape[0])!=(config["intrinsics"]["required_width"],config["intrinsics"]["required_height"]):
            raise RuntimeError(f"FORMAL_RESOLUTION_REQUIRED: got {frame.shape[1]}x{frame.shape[0]}")
        intrinsics=load_phase2b_intrinsics(PROJECT_ROOT,config,frame.shape[1],frame.shape[0])
        recorder.set_metadata(intrinsics_mode=intrinsics.mode,
            calibration_id=config["intrinsics"]["calibration_id"])
        while ok:
            target,distance,hand_position,repeat=plan[index]
            now=time.perf_counter(); probe=hand.process(frame); depth_map=depth.process(frame)
            result=core.process(probe.world_landmarks_m,probe.pixel_landmarks,depth_map,intrinsics,
                                depth_cal.correct if depth_cal.calibrated else None)
            hits={}
            for prefix,name in CANDIDATE_MAP.items():
                candidate=result.candidates.get(name)
                hits[prefix]=evaluate_camera_candidate(prefix,candidate,targets,targets[target],
                    formal["target_width_mm"],formal["target_height_mm"],formal["intersection_epsilon"])
                if not probe.is_pointing and hits[prefix].in_target_range:
                    hits[prefix]=replace(hits[prefix],in_target_range=False,status="NOT_POINTING",
                                         pred_region=None,error_mm=None)
            b=result.candidates.get("B_RELATIVE"); c=result.candidates.get("C_AXIS_FIT")
            if not probe.is_pointing:
                selected_name,selected_hit="INVALID",None
            elif args.direction_mode=="baseline":
                selected_name,selected_hit=("A_BASELINE",hits["A"]) if hits["A"].in_target_range else ("INVALID",None)
            elif args.direction_mode=="relative":
                selected_name,selected_hit=("B_RELATIVE",hits["B"]) if hits["B"].in_target_range else ("INVALID",None)
            elif args.direction_mode=="axisfit":
                selected_name,selected_hit=("C_AXIS_FIT",hits["C"]) if hits["C"].in_target_range else ("INVALID",None)
            else:
                selected_name,selected_hit=select_quality_fallback(hits["C"],c.quality if c else "INVALID",
                    hits["B"],b.quality if b else "INVALID",
                    tuple(config["quality_fallback"]["c_accepted_quality"]),
                    tuple(config["quality_fallback"]["b_accepted_quality"]))
            context={"session_id":recorder.session_id,
                     "trial_id":f"{args.protocol}_{index+1:03d}_{target}_{repeat:02d}",
                     "protocol":args.protocol,"gt_target":target,"gt_xy":targets[target],
                     "distance_cm":distance,"hand_position":hand_position,
                     "calibration_id":config["intrinsics"]["calibration_id"],"mirror_display":mirror}
            if state=="PREPARING" and now>=deadline:
                state="COLLECTING"; deadline=now+float(formal["collect_seconds"]); rows=[]
            if state=="COLLECTING":
                rows.append(_frame_row(context,probe,result,hits,selected_name,selected_hit,depth.mode,mirror))
                if now>=deadline:
                    trial=recorder.save_trial(rows,context,targets,formal["minimum_valid_frames"])
                    message=(f"Saved {context['trial_id']} | B {trial['B_failure_reason']} | "
                             f"C {trial['C_failure_reason']}")
                    pending_screenshot=context["trial_id"]
                    state="IDLE"; rows=[]
                    if index+1<len(plan): index+=1
                    else: protocol_complete=True
            display=cv2.flip(frame,1) if mirror else frame.copy(); _draw_hand(display,probe,mirror)
            _draw_camera_rays(display,result,intrinsics,mirror)
            cv2.rectangle(display,(0,0),(display.shape[1],300),(20,20,20),-1)
            remaining=max(0.0,deadline-now) if state!="IDLE" else 0.0
            _text(display,"PHASE 2B - CALIBRATED Z=0",(12,28),(0,255,255),.72,2)
            _text(display,f"{args.protocol.upper()} {index+1}/{len(plan)} | {state} {remaining:.1f}s | Mirror {'ON' if mirror else 'OFF'} (M) | SPACE | Q",(12,54),(220,220,220),.43)
            _text(display,f"GT {target} ({targets[target][0]:+.0f},{targets[target][1]:+.0f})mm | distance {distance:.0f}cm | hand {hand_position}",(12,80))
            _text(display,f"Intrinsics {intrinsics.mode} | PnP {result.pnp_status} RMSE {_fmt(result.pnp_rmse_px)}px",(12,105))
            _text(display,f"V2 anchor {result.anchor_v2.status} | raw/filter Z {_fmt(result.raw_anchor_depth_m,3)}/{_fmt(result.filtered_anchor_depth_m,3)}m",(12,130))
            for line,prefix in enumerate(("A","B","C")):
                candidate=result.candidates.get(CANDIDATE_MAP[prefix]); hit=hits[prefix]
                direction=candidate.smoothed_direction_camera if candidate else None
                _text(display,f"{prefix}: q={candidate.quality if candidate else 'INVALID'} dz={_fmt(direction[2] if direction is not None else None,3)} hit={hit.status} xy={_fmt(hit.hit_eval_mm[0] if hit.hit_eval_mm is not None else None)},{_fmt(hit.hit_eval_mm[1] if hit.hit_eval_mm is not None else None)} err={_fmt(hit.error_mm)}mm",
                      (12,160+line*25),COLORS[prefix])
            _text(display,f"Fallback selected: {selected_name} | Gesture {'POINTING' if probe.is_pointing else 'NOT_POINTING'}",(12,245))
            _text(display,message,(12,274),(0,200,255),.43,1)
            _target_panel(display,targets,target,hits)
            if pending_screenshot:
                screenshot=assert_safe_path(recorder.screenshot_dir/f"{pending_screenshot}.png")
                _write_screenshot(screenshot,display); pending_screenshot=None
                if protocol_complete: message="Protocol complete. Press Q to save report."
            frame_count+=1
            if not args.headless:
                cv2.imshow("Phase 2B Z=0 Evaluation",display); key=cv2.waitKey(1)&0xff
                if key in (27,ord('q'),ord('Q')): break
                if key in (ord('m'),ord('M')) and state=="IDLE": mirror=not mirror
                elif key==32 and state=="IDLE" and not protocol_complete:
                    core.reset(); state="PREPARING"; deadline=now+float(formal["prepare_seconds"])
                    message=f"Prepare: point to {target}; hand position {hand_position}"
            if args.max_frames and frame_count>=args.max_frames: break
            ok,frame=camera.read()
    finally:
        hand.close(); camera.release(); cv2.destroyAllWindows(); summary=recorder.close()
        print("Phase 2B output:",recorder.session_dir)
        print("Phase 2B status:",summary["status"])


if __name__=="__main__": main()
