from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from evaluation.coordinate_adapter import (KEYPAD_TARGETS, REGION_ORDER,
                                             configured_targets, mirror_display_x,
                                             nearest_region)
from evaluation.eval_pipeline import Z0EvaluationPipeline
from evaluation.trial_recorder import ActiveTrial, TrialRecorder
from evaluation.visualization import draw_hand_overlay, write_png
from src.camera.camera import Camera
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path


LOGICAL_GRID = {
    "LEFT_UP": (-1, 1), "UP": (0, 1), "RIGHT_UP": (1, 1),
    "LEFT": (-1, 0), "CENTER": (0, 0), "RIGHT": (1, 0),
    "LEFT_DOWN": (-1, -1), "DOWN": (0, -1), "RIGHT_DOWN": (1, -1),
}


def _text(image, value, xy, color=(255,255,255), scale=.5, thickness=1):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def mirror_sanity_check(targets):
    synthetic = targets if len(targets) == 9 else {
        name: (x*150.0, y*100.0) for name, (x,y) in LOGICAL_GRID.items()}
    for name in ("LEFT", "RIGHT", "LEFT_UP", "RIGHT_UP"):
        point = synthetic[name]
        before = nearest_region(*point, synthetic)
        mirror_display_x(10, 100, False); mirror_display_x(10, 100, True)
        after = nearest_region(*point, synthetic)
        if before != after:
            return "MIRROR_COORDINATE_BUG"
    return "PASS"


def draw_target_panel(image, targets, gt_name, hit_mm, pred_region):
    height, width = image.shape[:2]
    panel_w, panel_h = min(330, width//3), min(260, height//2)
    x0, y0 = width-panel_w-12, 76
    cv2.rectangle(image, (x0,y0), (x0+panel_w,y0+panel_h), (25,25,25), -1)
    cv2.rectangle(image, (x0,y0), (x0+panel_w,y0+panel_h), (210,210,210), 1)
    _text(image, "Z=0 TARGET PLANE", (x0+10,y0+22), (0,255,255), .5)
    configured = targets or {name:(x*150,y*100) for name,(x,y) in LOGICAL_GRID.items()}
    max_x = max(max(abs(v[0]) for v in configured.values()), 1.0)
    max_y = max(max(abs(v[1]) for v in configured.values()), 1.0)

    def to_panel(point):
        px = x0+panel_w//2 + int(point[0]/max_x*(panel_w*.38))
        py = y0+panel_h//2 - int(point[1]/max_y*(panel_h*.32))
        return px,py
    for name, logical in LOGICAL_GRID.items():
        point = targets.get(name, (logical[0]*max_x,logical[1]*max_y))
        p = to_panel(point)
        color = (120,120,120) if name not in targets else (220,220,220)
        _text(image, name.replace("RIGHT","R").replace("LEFT","L").replace("DOWN","D").replace("UP","U").replace("CENTER","C"),
              (p[0]-12,p[1]+5),color,.35)
    if gt_name in targets:
        cv2.circle(image, to_panel(targets[gt_name]), 7, (0,220,0), -1)
    if hit_mm is not None:
        px,py = to_panel(hit_mm)
        cv2.line(image,(px-7,py-7),(px+7,py+7),(0,0,255),3)
        cv2.line(image,(px-7,py+7),(px+7,py-7),(0,0,255),3)
    _text(image, "GT: green dot | Prediction: red X", (x0+8,y0+panel_h-10), (200,200,200), .35)
    return image


def render_ui(frame, result, state, selected_target, gt_xy, distance_cm, hand_position,
              mirror_display, remaining, targets, message, workflow="MANUAL"):
    display = cv2.flip(frame,1) if mirror_display else frame.copy()
    width = frame.shape[1]
    direction_visible = draw_hand_overlay(display,result,mirror_display,width)
    for name, point in result.points_2d.items():
        if point is None: continue
        x = int(round(mirror_display_x(point[0],width,mirror_display))); y=int(round(point[1]))
        _text(display,name.upper(),(x+7,y-7),(0,255,255),.4)
    cv2.rectangle(display,(0,0),(display.shape[1],68),(20,20,20),-1)
    _text(display,"Z=0 POINTING EVALUATION",(12,30),(0,255,255),.78,2)
    state_label={"PREPARING":"READY","COLLECTING":"RECORDING"}.get(state,state)
    _text(display,f"{state_label}: {remaining:.1f}s | {workflow} | M mirror | H hand-pos | D distance | SPACE trial | Q quit",
          (12,56),(220,220,220),.43)
    pred = result.pred_region or "N/A"
    hit = result.hit_eval_mm
    error = float(np.linalg.norm(hit-np.asarray(gt_xy))) if hit is not None and gt_xy is not None else None
    points=result.points_3d
    def z_value(name):
        point=points.get(name)
        return "N/A" if point is None else f"{float(point[2]):+.4f}m"
    raw=result.raw_direction
    raw_text=(f"dx={raw[0]:+.4f} dy={raw[1]:+.4f} dz={raw[2]:+.4f}"
              if raw is not None else "dx/dy/dz=N/A")
    angle="N/A" if result.angle_to_camera_axis_deg is None else f"{result.angle_to_camera_axis_deg:.1f} deg"
    norm="N/A" if result.direction_norm is None else f"{result.direction_norm:.4f} m"
    flags=", ".join(result.sanity_flags) if result.sanity_flags else "NONE"
    raw_hit=result.raw_hit_eval_mm
    lines = [
        f"GT TARGET: {selected_target}   PREDICTED: {pred}",
        f"GT XY: ({gt_xy[0]:+.0f}, {gt_xy[1]:+.0f}) mm" if gt_xy else "GT XY: NOT CONFIGURED",
        f"PRED XY: ({hit[0]:+.1f}, {hit[1]:+.1f}) mm" if hit is not None else "PRED XY: N/A",
        f"ERROR: {error:.1f} mm" if error is not None else "ERROR: N/A",
        f"Gesture: {result.gesture_label} ({result.gesture_confidence:.2f})",
        f"MCP Z={z_value('mcp')}  PIP Z={z_value('pip')}  DIP Z={z_value('dip')}  TIP Z={z_value('tip')}",
        f"TIP-PIP: {raw_text}",
        f"Direction Norm: {norm}  Angle to Camera Axis: {angle}",
        f"Direction Quality: {result.direction_quality}  Ray Status: {result.ray_status}",
        f"Depth Order: {result.depth_order_status}  Sanity: {flags}",
        f"Current Direction: {'VISIBLE' if direction_visible else 'UNAVAILABLE'}",
        f"Z=0 Intersection: {'YES' if result.intersection_valid else 'NO'} ({result.intersection_status})",
        (f"Raw Hit: ({raw_hit[0]:+.1f}, {raw_hit[1]:+.1f}) mm" if raw_hit is not None else "Raw Hit: N/A"),
        f"Hand Distance GT: {distance_cm} cm  Hand Position: {hand_position}",
        f"FPS: {result.fps:.1f}  Depth: {result.depth_mode}",
        f"Camera: {result.intrinsics_mode}  Mirror Display: {'ON' if mirror_display else 'OFF'}",
    ]
    for index,line in enumerate(lines): _text(display,line,(12,92+index*21),(255,255,255),.43)
    if state in ("PREPARING","COLLECTING"):
        _text(display,f"Please point to {selected_target}",(display.shape[1]//3,display.shape[0]-45),(0,255,255),.75,2)
    if message: _text(display,message,(12,display.shape[0]-18),(0,180,255),.5,2)
    return draw_target_panel(display,targets,selected_target,hit,pred)


def main():
    parser=argparse.ArgumentParser(description="Independent Z=0 camera-plane pointing evaluation")
    parser.add_argument("--depth",choices=["auto","metric","approximate"],default=None)
    parser.add_argument("--distance-cm",type=float,default=None)
    parser.add_argument("--hand-position",choices=["LEFT","CENTER","RIGHT","UP","DOWN"],default=None)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--mode",choices=["quick","full","manual"],default="quick")
    args=parser.parse_args()
    project_cfg=load_yaml(PROJECT_ROOT/"configs/default.yaml")
    eval_cfg=load_yaml(PROJECT_ROOT/"evaluation/configs/z0_eval.yaml")
    raw_targets=load_yaml(PROJECT_ROOT/eval_cfg["targets_file"])
    targets=configured_targets(raw_targets)
    if args.depth: eval_cfg["depth"]["backend"]=args.depth
    distance=float(args.distance_cm or eval_cfg["default_distance_cm"])
    hand_positions=eval_cfg["hand_positions"]; hand_position=args.hand_position or eval_cfg["default_hand_position"]
    mirror=bool(eval_cfg["mirror_display"])
    mode_cfg=eval_cfg["evaluation_modes"].get(args.mode,{})
    plan=[(float(distance_value),target,repeat+1)
          for distance_value in mode_cfg.get("distances_cm",[])
          for target in mode_cfg.get("targets",[])
          for repeat in range(int(mode_cfg.get("trials_per_target",0)))]
    plan_index=0
    selected=plan[0][1] if plan else "CENTER"
    if plan and args.distance_cm is None: distance=plan[0][0]
    camera_cfg=project_cfg["camera"]
    camera=Camera(camera_cfg["index"],camera_cfg["width"],camera_cfg["height"],camera_cfg["fps"])
    recorder=TrialRecorder(PROJECT_ROOT/eval_cfg["output_root"])
    pipeline=None; active=None; state="IDLE"; state_deadline=0.0; trial_number=0; screenshot_pending=None
    advance_pending=False
    mirror_check=mirror_sanity_check(targets); message=""
    fps=0.0; previous=time.perf_counter(); frame_count=0
    try:
        camera.open(); ok,frame=camera.read()
        if not ok: raise RuntimeError("Camera opened but returned no frame")
        pipeline=Z0EvaluationPipeline(project_cfg,eval_cfg,PROJECT_ROOT,(frame.shape[1],frame.shape[0]),targets)
        depth_label=pipeline.depth.mode+("_CORRECTED" if pipeline.depth_cal.calibrated else "_UNCALIBRATED")
        recorder.set_metadata(camera=f"Camera index {camera_cfg['index']}",
                              resolution=f"{frame.shape[1]} x {frame.shape[0]}",fps=camera_cfg["fps"],
                              depth_mode=depth_label,intrinsics_mode=pipeline.intrinsics.mode,
                              mirror=f"Initial {'ON' if mirror else 'OFF'}; display-only, per-trial recorded",
                              test_distance=args.mode,
                              target_plane="A4 210 x 297 mm, camera lens at CENTER")
        missing=[name for name in REGION_ORDER if name not in targets]
        if missing: message="TARGET CONFIG INCOMPLETE: fill z0_targets.yaml before formal trials"
        while ok:
            now=time.perf_counter()
            result=pipeline.process(frame,datetime.now().isoformat(),fps)
            current=time.perf_counter(); fps=1/max(current-previous,1e-6); previous=current; result.fps=fps
            if state=="PREPARING" and current>=state_deadline:
                trial_number+=1
                context={"trial_id":f"trial_{trial_number:04d}","gt_target":selected,
                         "gt_xy":targets[selected],"distance_cm":distance,"hand_position":hand_position,
                         "mirror_display":mirror}
                active=ActiveTrial(context); state="COLLECTING"; state_deadline=current+float(eval_cfg["collect_seconds"])
            if state=="COLLECTING":
                active.add(result)
                if current>=state_deadline:
                    trial=recorder.save_trial(active,targets,eval_cfg["stability"],eval_cfg["minimum_valid_frames"],eval_cfg["direction_quality"])
                    message=(f"Saved {trial['trial_id']}: keypoints={trial['keypoint_stability']} "
                             f"direction={trial['direction_quality']} pred={trial['pred_region']} "
                             f"failure={trial['failure_reason']}")
                    screenshot_pending=trial["trial_id"]
                    state="IDLE"; active=None
                    advance_pending=True
            remaining=max(0.0,state_deadline-current) if state!="IDLE" else 0.0
            workflow=(f"{args.mode.upper()} {min(plan_index+1,len(plan))}/{len(plan)}" if plan else "MANUAL")
            display=render_ui(frame,result,state,selected,targets.get(selected),distance,hand_position,
                              mirror,remaining,targets,message,workflow)
            if screenshot_pending:
                screenshot_path=assert_safe_path(recorder.screenshot_dir/f"{screenshot_pending}.png")
                if not write_png(screenshot_path,display):
                    raise RuntimeError(f"Failed to save trial screenshot: {screenshot_path}")
                screenshot_pending=None
                if advance_pending and plan and plan_index+1<len(plan):
                    plan_index+=1; distance,selected,_=plan[plan_index]
                elif advance_pending and plan and plan_index+1==len(plan):
                    message=f"{args.mode.upper()} evaluation complete ({len(plan)} trials). Q to generate report."
                advance_pending=False
            frame_count+=1
            if not args.headless:
                cv2.imshow("Z=0 Pointing Evaluation",display)
                key=cv2.waitKey(1)&0xFF
                if key in (27,ord('q'),ord('Q')): break
                if key in KEYPAD_TARGETS and state=="IDLE":
                    selected=KEYPAD_TARGETS[key]; plan=[]; message=f"GT selected: {selected} (MANUAL)"
                elif key in (ord('m'),ord('M')) and state=="IDLE": mirror=not mirror; message="Display mirror toggled; internal geometry unchanged"
                elif key in (ord('h'),ord('H')) and state=="IDLE":
                    hand_position=hand_positions[(hand_positions.index(hand_position)+1)%len(hand_positions)]
                elif key in (ord('d'),ord('D')) and state=="IDLE":
                    values=[float(v) for v in eval_cfg["test_distances_cm"]]
                    distance=values[(values.index(distance)+1)%len(values)] if distance in values else values[0]
                elif key==32 and state=="IDLE":
                    if selected not in targets: message=f"GT {selected} is not measured in z0_targets.yaml"
                    else: state="PREPARING"; state_deadline=current+float(eval_cfg["prepare_seconds"]); message="Hold pointing pose"
            if args.max_frames and frame_count>=args.max_frames: break
            ok,frame=camera.read()
    finally:
        if pipeline: pipeline.close()
        camera.release(); cv2.destroyAllWindows(); recorder.close(mirror_check)
        print("Z=0 evaluation output:",recorder.session_dir)
        print("Mirror sanity check:",mirror_check)


if __name__=="__main__": main()
