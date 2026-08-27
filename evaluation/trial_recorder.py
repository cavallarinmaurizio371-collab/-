from __future__ import annotations

import csv
import json
import os
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from evaluation.coordinate_adapter import REGION_ORDER, nearest_region
from evaluation.diagnostics import classify_keypoint_stability
from evaluation.metrics import build_summary, confusion_matrix
from evaluation.stability_analyzer import analyze_stability
from src.safety.path_guard import PROJECT_ROOT, safe_mkdir, safe_open


FRAME_FIELDS = [
    "trial_id", "timestamp", "gt_target", "gt_x_mm", "gt_y_mm",
    "measured_hand_distance_cm", "hand_position_in_frame", "hand_detected",
    "gesture_label", "gesture_confidence", "mcp_2d", "pip_2d", "dip_2d", "tip_2d",
    "mcp_xyz", "pip_xyz", "dip_xyz", "tip_xyz", "pointing_direction_xyz",
    "raw_direction_xyz", "direction_norm", "angle_to_camera_axis_deg", "direction_quality",
    "ray_status", "depth_order_status", "sanity_flags",
    "ray_valid", "intersection_valid", "intersection_status", "pred_x_mm", "pred_y_mm",
    "raw_pred_x_mm", "raw_pred_y_mm",
    "pred_region", "region_correct", "error_x_mm", "error_y_mm", "radial_error_mm",
    "fps", "depth_mode", "camera_intrinsics_mode", "mirror_display_enabled",
]

TRIAL_FIELDS = [
    "trial_id", "timestamp", "gt_target", "gt_x_mm", "gt_y_mm",
    "measured_hand_distance_cm", "hand_position_in_frame", "frame_count", "hand_detected_frames",
    "pointing_frames", "valid_direction_frames", "valid_intersection_frames", "intersection_valid",
    "pred_x_mm", "pred_y_mm", "pred_region", "region_correct", "error_x_mm", "error_y_mm",
    "radial_error_mm", "tip_pixel_std", "pip_pixel_std", "dip_pixel_std", "tip_xyz_std_m",
    "pip_xyz_std_m", "dip_xyz_std_m", "direction_angle_std_deg", "direction_dz_negative_rate",
    "tip_z_std_m", "pip_z_std_m", "dip_z_std_m", "tip_pip_depth_order_consistency",
    "hit_x_std_mm", "hit_y_std_mm", "hit_radial_std_mm", "stability_status",
    "keypoint_stability", "raw_frame_direction", "median_direction", "stable_frame_direction",
    "direction_quality", "failure_reason",
    "depth_mode", "camera_intrinsics_mode", "mirror_display_enabled",
]


def _encoded(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_to_row(frame, context):
    gt_x, gt_y = context["gt_xy"]
    pred_x, pred_y = frame.hit_eval_mm if frame.hit_eval_mm is not None else (None, None)
    raw_hit = getattr(frame, "raw_hit_eval_mm", None)
    raw_x, raw_y = raw_hit if raw_hit is not None else (None, None)
    error_x = pred_x-gt_x if pred_x is not None else None
    error_y = pred_y-gt_y if pred_y is not None else None
    radial = float(np.hypot(error_x, error_y)) if error_x is not None else None
    row = {
        "trial_id": context["trial_id"], "timestamp": frame.timestamp,
        "gt_target": context["gt_target"], "gt_x_mm": gt_x, "gt_y_mm": gt_y,
        "measured_hand_distance_cm": context["distance_cm"],
        "hand_position_in_frame": context["hand_position"], "hand_detected": frame.hand_detected,
        "gesture_label": frame.gesture_label, "gesture_confidence": frame.gesture_confidence,
        **{f"{name}_2d": frame.points_2d.get(name) for name in ("mcp","pip","dip","tip")},
        **{f"{name}_xyz": frame.points_3d.get(name) for name in ("mcp","pip","dip","tip")},
        "pointing_direction_xyz": frame.baseline_direction, "ray_valid": frame.ray_valid,
        "raw_direction_xyz":getattr(frame,"raw_direction",None),
        "direction_norm":getattr(frame,"direction_norm",None),
        "angle_to_camera_axis_deg":getattr(frame,"angle_to_camera_axis_deg",None),
        "direction_quality":getattr(frame,"direction_quality","INVALID"),
        "ray_status":getattr(frame,"ray_status","INVALID"),
        "depth_order_status":getattr(frame,"depth_order_status","MISSING_KEYPOINT"),
        "sanity_flags":getattr(frame,"sanity_flags",[]),
        "intersection_valid": frame.intersection_valid, "intersection_status": frame.intersection_status,
        "pred_x_mm": pred_x, "pred_y_mm": pred_y, "pred_region": frame.pred_region,
        "raw_pred_x_mm":raw_x,"raw_pred_y_mm":raw_y,
        "region_correct": frame.pred_region == context["gt_target"] if frame.pred_region else False,
        "error_x_mm": error_x, "error_y_mm": error_y, "radial_error_mm": radial,
        "fps": frame.fps, "depth_mode": frame.depth_mode,
        "camera_intrinsics_mode": frame.intrinsics_mode,
        "mirror_display_enabled": context["mirror_display"],
    }
    return {key: _encoded(row.get(key)) for key in FRAME_FIELDS}


@dataclass
class ActiveTrial:
    context: dict
    frames: list = field(default_factory=list)

    def add(self, frame):
        self.frames.append(frame)

    @staticmethod
    def _direction_aggregate(frames, quality_config, angle_std):
        directions=[np.asarray(frame.baseline_direction,dtype=float) for frame in frames
                    if frame.baseline_direction is not None and np.all(np.isfinite(frame.baseline_direction))]
        if not directions:
            return None,None,None,"INVALID"
        normalized=[d/np.linalg.norm(d) for d in directions if np.linalg.norm(d)>1e-9]
        if not normalized:
            return None,None,None,"INVALID"
        directions=np.stack(normalized)
        raw=directions[-1]
        median=np.median(directions,axis=0); median/=np.linalg.norm(median)
        angles=np.degrees(np.arccos(np.clip(directions@median,-1,1)))
        stable_set=directions[angles<=float(quality_config["marginal_angle_std_deg"])]
        stable=np.mean(stable_set,axis=0) if len(stable_set) else median.copy()
        stable/=np.linalg.norm(stable)
        if angle_std is not None and angle_std>float(quality_config["marginal_angle_std_deg"]): quality="UNSTABLE"
        elif median[2]>0: quality="AWAY"
        elif abs(median[2])<float(quality_config["near_parallel_abs_dz"]): quality="NEAR_PARALLEL"
        elif abs(median[2])>=float(quality_config["good_min_toward_abs_dz"]) and (angle_std is None or angle_std<=float(quality_config["good_angle_std_deg"])): quality="GOOD"
        else: quality="MARGINAL"
        return raw,median,stable,quality

    @staticmethod
    def _failure_reason(frames, prediction, pred_region, gt_target, stability,
                        keypoint_quality, direction_quality, quality_config):
        detected=sum(frame.hand_detected for frame in frames)
        if detected==0: return "NO_HAND"
        pointing=sum(frame.gesture_label=="POINTING" for frame in frames)
        if pointing/detected<float(quality_config["minimum_pointing_frame_rate"]): return "NOT_POINTING"
        if keypoint_quality=="BAD": return "KEYPOINT_UNSTABLE"
        flags=[flag for frame in frames for flag in getattr(frame,"sanity_flags",[])]
        if "DIRECTION_SIGN_INCONSISTENT" in flags: return "DIRECTION_SIGN_INCONSISTENT"
        consistency=stability.get("tip_pip_depth_order_consistency")
        if consistency is not None and consistency<float(quality_config["minimum_depth_order_consistency"]): return "DEPTH_ORDER_INCONSISTENT"
        if direction_quality=="UNSTABLE": return "KEYPOINT_UNSTABLE"
        if direction_quality=="AWAY": return "AWAY_FROM_CAMERA"
        if direction_quality=="NEAR_PARALLEL": return "NEAR_PARALLEL"
        if any(frame.intersection_status=="OUT_OF_TARGET_RANGE" for frame in frames) and prediction is None: return "OUT_OF_TARGET_RANGE"
        if prediction is None: return "INVALID"
        return "VALID_CORRECT_REGION" if pred_region==gt_target else "VALID_WRONG_REGION"

    def finalize(self, targets, stability_thresholds, minimum_valid_frames=3, direction_config=None):
        direction_config=direction_config or {
            "near_parallel_abs_dz":.15,"good_min_toward_abs_dz":.55,
            "good_angle_std_deg":8,"marginal_angle_std_deg":15,
            "minimum_pointing_frame_rate":.5,"minimum_depth_order_consistency":.6}
        gt_x, gt_y = self.context["gt_xy"]
        hits = [frame.hit_eval_mm for frame in self.frames if frame.hit_eval_mm is not None]
        enough = len(hits) >= int(minimum_valid_frames)
        prediction = np.median(np.stack(hits), axis=0) if enough else None
        pred_region = nearest_region(*prediction, targets) if prediction is not None else None
        error_x = float(prediction[0]-gt_x) if prediction is not None else None
        error_y = float(prediction[1]-gt_y) if prediction is not None else None
        stability = analyze_stability(self.frames, stability_thresholds)
        keypoint_quality=classify_keypoint_stability(stability,stability_thresholds)
        raw_direction,median_direction,stable_direction,direction_quality=self._direction_aggregate(
            self.frames,direction_config,stability.get("direction_angle_std_deg"))
        failure_reason=self._failure_reason(self.frames,prediction,pred_region,self.context["gt_target"],
                                            stability,keypoint_quality,direction_quality,direction_config)
        row = {
            "trial_id": self.context["trial_id"], "timestamp": datetime.now().isoformat(),
            "gt_target": self.context["gt_target"], "gt_x_mm": gt_x, "gt_y_mm": gt_y,
            "measured_hand_distance_cm": self.context["distance_cm"],
            "hand_position_in_frame": self.context["hand_position"], "frame_count": len(self.frames),
            "hand_detected_frames": sum(frame.hand_detected for frame in self.frames),
            "pointing_frames": sum(frame.gesture_label == "POINTING" for frame in self.frames),
            "valid_direction_frames": sum(frame.ray_valid for frame in self.frames),
            "valid_intersection_frames": len(hits), "intersection_valid": enough,
            "pred_x_mm": float(prediction[0]) if prediction is not None else None,
            "pred_y_mm": float(prediction[1]) if prediction is not None else None,
            "pred_region": pred_region, "region_correct": pred_region == self.context["gt_target"],
            "error_x_mm": error_x, "error_y_mm": error_y,
            "radial_error_mm": float(np.hypot(error_x,error_y)) if error_x is not None else None,
            "depth_mode": self.frames[-1].depth_mode if self.frames else None,
            "camera_intrinsics_mode": self.frames[-1].intrinsics_mode if self.frames else None,
            "mirror_display_enabled": self.context["mirror_display"],
            "keypoint_stability":keypoint_quality,"raw_frame_direction":raw_direction,
            "median_direction":median_direction,"stable_frame_direction":stable_direction,
            "direction_quality":direction_quality,"failure_reason":failure_reason,
            **stability,
        }
        row.update({"tip_pixel_std": row.pop("tip_pixel_std"),
                    "pip_pixel_std": row.pop("pip_pixel_std"),
                    "dip_pixel_std": row.pop("dip_pixel_std")})
        return {key: _encoded(row.get(key)) for key in TRIAL_FIELDS}


class TrialRecorder:
    def __init__(self, output_root):
        root = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
        self.session_dir = safe_mkdir(root / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"))
        self.screenshot_dir = safe_mkdir(self.session_dir / "screenshots")
        self.optional_video_dir = safe_mkdir(self.session_dir / "optional_video")
        self.trials_path, self.frames_path = self.session_dir/"trials.csv", self.session_dir/"frames.csv"
        self.trial_file = safe_open(self.trials_path, "w", newline="", encoding="utf-8")
        self.frame_file = safe_open(self.frames_path, "w", newline="", encoding="utf-8")
        self.trial_writer = csv.DictWriter(self.trial_file, fieldnames=TRIAL_FIELDS)
        self.frame_writer = csv.DictWriter(self.frame_file, fieldnames=FRAME_FIELDS)
        self.trial_writer.writeheader(); self.frame_writer.writeheader()
        self.trials, self.frames = [], []
        self.metadata = {}

    def set_metadata(self, **metadata):
        self.metadata.update(metadata)

    def save_trial(self, active, targets, thresholds, minimum_valid_frames, direction_config=None):
        for frame in active.frames:
            row = frame_to_row(frame, active.context)
            self.frame_writer.writerow(row); self.frames.append(row)
        trial = active.finalize(targets, thresholds, minimum_valid_frames, direction_config)
        self.trial_writer.writerow(trial); self.trials.append(trial)
        self.trial_file.flush(); self.frame_file.flush()
        return trial

    def close(self, mirror_check="PASS"):
        if self.trial_file.closed:
            return
        self.trial_file.close(); self.frame_file.close()
        summary = build_summary(self.trials, self.frames, mirror_check)
        with safe_open(self.session_dir/"summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        matrix = confusion_matrix(self.trials)
        with safe_open(self.session_dir/"region_confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["GT/PRED", *REGION_ORDER])
            for gt in REGION_ORDER: writer.writerow([gt, *[matrix[gt][pred] for pred in REGION_ORDER]])
        self._write_confusion_png(matrix)
        self._write_markdown(summary)
        self._write_failure_analysis(summary)

    def _write_confusion_png(self, matrix):
        os.environ.setdefault("MPLCONFIGDIR", str(self.session_dir/"matplotlib_cache"))
        os.environ.setdefault("WINDIR", r"C:\Windows")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        values=np.asarray([[matrix[gt][pred] for pred in REGION_ORDER] for gt in REGION_ORDER])
        figure=Figure(figsize=(9,8),dpi=130); FigureCanvasAgg(figure)
        axis=figure.subplots(); image=axis.imshow(values,cmap="Blues",vmin=0)
        axis.set_xticks(range(9),REGION_ORDER,rotation=45,ha="right",fontsize=7)
        axis.set_yticks(range(9),REGION_ORDER,fontsize=7)
        axis.set_xlabel("Predicted region"); axis.set_ylabel("Ground truth")
        axis.set_title("Z=0 Region Confusion Matrix")
        for row in range(9):
            for column in range(9):
                axis.text(column,row,str(values[row,column]),ha="center",va="center",
                          color="white" if values[row,column]>values.max()/2 and values.max() else "black")
        figure.colorbar(image,ax=axis); figure.tight_layout()
        buffer=BytesIO(); figure.savefig(buffer,format="png")
        with safe_open(self.session_dir/"region_confusion_matrix.png","wb") as handle:
            handle.write(buffer.getvalue())

    def _write_markdown(self, summary):
        def fmt(value, suffix=""):
            return "N/A" if value is None else f"{value:.3f}{suffix}" if isinstance(value,float) else str(value)
        warning = ("当前毫米级误差属于未完成真实深度校准和相机内参标定条件下的初步结果，"
                   "不代表最终物理测量精度；区域准确率、方向有效率和关键点稳定性仍可用于模型评测。")
        environment={"Camera":self.metadata.get("camera","Camera index 0"),
                     "Resolution":self.metadata.get("resolution","N/A"),
                     "FPS":self.metadata.get("fps","N/A"),
                     "Depth Mode":self.metadata.get("depth_mode","N/A"),
                     "Intrinsics Mode":self.metadata.get("intrinsics_mode","N/A"),
                     "Mirror":self.metadata.get("mirror","N/A"),
                     "Test Distance":self.metadata.get("test_distance","N/A"),
                     "Target Plane":self.metadata.get("target_plane","A4 210 x 297 mm, camera centered")}
        rows=[("Hand Detection Rate",summary["hand_detection_rate"]),
              ("Pointing Recognition Rate",summary["pointing_recognition_rate"]),
              ("Valid Direction Rate",summary["valid_3d_direction_rate"]),
              ("Valid Intersection Rate",summary["valid_z0_intersection_rate"]),
              ("Region Accuracy",summary["region_accuracy"]),
              ("Mean Error (mm)",summary["mean_radial_error_mm"]),
              ("Median Error (mm)",summary["median_radial_error_mm"]),
              ("P90 Error (mm)",summary["p90_radial_error_mm"])]
        lines=["# Z=0 正指评测报告","","## 1. 评测目的","",
               "验证食指正对摄像头时的关键点稳定性、三维方向、Z=0 落点和九区域预测。","",
               "## 2. 环境","","| 项目 | 值 |","|---|---|",
               *[f"| {key} | {value} |" for key,value in environment.items()],"",
               "## 3. 总体结果","","| 指标 | 结果 |","|---|---|",
               *[f"| {name} | {fmt(value)} |" for name,value in rows],"",
               f"> {warning}","", "## 4. 按目标区域","","```json",
               json.dumps(summary["by_gt_target"],indent=2,ensure_ascii=False),"```","",
               "## 5. 按距离","","```json",json.dumps(summary["by_distance_cm"],indent=2,ensure_ascii=False),"```","",
               "## 6. 按手位置","","```json",json.dumps(summary["by_hand_position"],indent=2,ensure_ascii=False),"```","",
               "### GT=CENTER 的手位置专项","","```json",json.dumps(summary["center_gt_by_hand_position"],indent=2,ensure_ascii=False),"```","",
               "## 7. 失败原因分析","", "详见 `failure_analysis.md`。所有结论均来自本 Session 的实际 Trial，不补写或删除失败样本。","",
               "## 坐标与方向审计","",
               "相机坐标为 +X 向原始图像右、+Y 向下、+Z 由摄像头指向用户/场景；深度越大表示越远。",
               "方向沿用原业务 `normalize(TIP_3D - PIP_3D)`。因此手在 Z>0 且正指摄像头时，理论上 TIP_Z < PIP_Z、dz < 0。",
               f"Mirror isolation check: {summary['mirror_check']}。显示镜像不进入 XYZ、方向、求交、GT 或误差计算。"]
        with safe_open(self.session_dir/"summary.md", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def _write_failure_analysis(self, summary):
        lines=["# Failure Analysis","",f"Total trials: {summary['total_trials']}","",
               "| Failure reason | Count | Rate |","|---|---:|---:|"]
        for reason, values in summary["failure_distribution"].items():
            rate=values["rate"]
            lines.append(f"| {reason} | {values['count']} | {'N/A' if rate is None else f'{rate:.1%}'} |")
        if not summary["failure_distribution"]:
            lines.append("| No trials recorded | 0 | N/A |")
        lines.extend(["","解释边界：DEPTH_ORDER_INCONSISTENT 表示单目深度把 TIP 判断得比 PIP 更远；"
                      "NEAR_PARALLEL 表示归一化 dz 过小；OUT_OF_TARGET_RANGE 表示求交有限但落点超出 A4。",
                      "未采集到的失败类型不会被推断为主要原因。"]) 
        with safe_open(self.session_dir/"failure_analysis.md","w",encoding="utf-8") as handle:
            handle.write("\n".join(lines))
