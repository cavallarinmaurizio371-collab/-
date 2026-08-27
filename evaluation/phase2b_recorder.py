from __future__ import annotations

import csv
import json
import os
from collections import Counter,defaultdict
from datetime import datetime
from io import BytesIO

import numpy as np

from evaluation.coordinate_adapter import REGION_ORDER,nearest_region
from evaluation.phase2b_metrics import (error_stats,landing_error,paired_candidate_metrics,
    point_jitter,region_accuracy,trial_hit_median,vector_jitter_deg)
from src.safety.path_guard import PROJECT_ROOT,safe_mkdir,safe_open


PREFIXES=("A","B","C")


def _encoded(value):
    if isinstance(value,np.ndarray): value=value.tolist()
    if isinstance(value,(list,tuple,dict)): return json.dumps(value,ensure_ascii=False)
    if isinstance(value,np.generic): return value.item()
    return value


def _rate(numerator,denominator):
    return float(numerator)/float(denominator) if denominator else None


def _finite(rows,key):
    return [float(row[key]) for row in rows if row.get(key) is not None and
            np.isfinite(float(row[key]))]


def _candidate_trial(rows,prefix,gt_target,gt_xy,targets,minimum_valid_frames):
    hits=[row.get(f"{prefix}_hit_xy_mm") for row in rows if row.get(f"{prefix}_in_target_range")]
    prediction,hit_jitter=trial_hit_median(hits,minimum_valid_frames)
    region=nearest_region(*prediction,targets) if prediction is not None else None
    directions=[row.get(f"{prefix}_direction") for row in rows]
    origins=[row.get(f"{prefix}_origin") for row in rows]
    error=landing_error(prediction,gt_xy)
    complete=sum(bool(row.get(f"{prefix}_complete_ray_valid")) for row in rows)
    intersections=sum(bool(row.get(f"{prefix}_hit_valid")) for row in rows)
    in_range=sum(bool(row.get(f"{prefix}_in_target_range")) for row in rows)
    if prediction is not None:
        failure="VALID_CORRECT_REGION" if region==gt_target else "VALID_WRONG_REGION"
    else:
        statuses=Counter(str(row.get(f"{prefix}_intersection_status") or "INVALID") for row in rows)
        failure=statuses.most_common(1)[0][0] if statuses else "INVALID"
    return {f"{prefix}_complete_ray_frames":complete,
            f"{prefix}_toward_frames":sum(bool(row.get(f"{prefix}_toward_camera")) for row in rows),
            f"{prefix}_quality_frames":sum(bool(row.get(f"{prefix}_quality_valid")) for row in rows),
            f"{prefix}_valid_intersection_frames":intersections,
            f"{prefix}_in_target_frames":in_range,
            f"{prefix}_hit_x_mm":float(prediction[0]) if prediction is not None else None,
            f"{prefix}_hit_y_mm":float(prediction[1]) if prediction is not None else None,
            f"{prefix}_region":region,
            f"{prefix}_region_correct":bool(region==gt_target) if region else False,
            f"{prefix}_error_mm":error,
            f"{prefix}_direction_jitter_deg":vector_jitter_deg(directions),
            f"{prefix}_origin_jitter_m":point_jitter(origins),
            f"{prefix}_hit_jitter_mm":hit_jitter,
            f"{prefix}_failure_reason":failure}


def finalize_trial(rows,context,targets,minimum_valid_frames):
    result={"session_id":context["session_id"],"trial_id":context["trial_id"],
            "timestamp":datetime.now().isoformat(),"protocol":context["protocol"],
            "gt_target":context["gt_target"],"gt_x_mm":context["gt_xy"][0],
            "gt_y_mm":context["gt_xy"][1],"distance_cm":context["distance_cm"],
            "hand_position":context["hand_position"],"frame_count":len(rows),
            "hand_detected_frames":sum(bool(row.get("hand_detected")) for row in rows),
            "pointing_frames":sum(bool(row.get("pointing_state")) for row in rows),
            "pnp_valid_frames":sum(bool(row.get("pnp_valid")) for row in rows),
            "pnp_rmse_px_median":float(np.median(values)) if
                (values:=_finite(rows,"pnp_rmse_px")) else None,
            "anchor_v2_valid_frames":sum(bool(row.get("anchor_v2_valid")) for row in rows),
            "anchor_mad_median_m":float(np.median(values)) if
                (values:=_finite(rows,"anchor_mad")) else None,
            "intrinsics_mode":rows[-1].get("intrinsics_mode") if rows else None,
            "calibration_id":context["calibration_id"],
            "mirror_display":context["mirror_display"]}
    for prefix in PREFIXES:
        result.update(_candidate_trial(rows,prefix,context["gt_target"],context["gt_xy"],
                                       targets,minimum_valid_frames))
    selected_hits=[row.get("selected_hit_xy_mm") for row in rows if row.get("selected_candidate")!="INVALID"]
    selected,selected_jitter=trial_hit_median(selected_hits,minimum_valid_frames)
    result["selected_candidate"]=(Counter(row["selected_candidate"] for row in rows
        if row.get("selected_candidate")!="INVALID").most_common(1)[0][0]
        if any(row.get("selected_candidate")!="INVALID" for row in rows) else "INVALID")
    result["selected_hit_x_mm"]=float(selected[0]) if selected is not None else None
    result["selected_hit_y_mm"]=float(selected[1]) if selected is not None else None
    result["selected_region"]=nearest_region(*selected,targets) if selected is not None else None
    result["selected_region_correct"]=(result["selected_region"]==context["gt_target"]
                                       if result["selected_region"] else False)
    result["selected_error_mm"]=landing_error(selected,context["gt_xy"])
    result["selected_hit_jitter_mm"]=selected_jitter
    return result


def _candidate_summary(trials,frames,prefix):
    detected=sum(bool(row.get("hand_detected")) for row in frames)
    complete=sum(bool(row.get(f"{prefix}_complete_ray_valid")) for row in frames)
    intersections=sum(bool(row.get(f"{prefix}_hit_valid")) for row in frames)
    in_range=sum(bool(row.get(f"{prefix}_in_target_range")) for row in frames)
    predicted=[row for row in trials if row.get(f"{prefix}_region")]
    return {"complete_ray_valid_rate":_rate(complete,detected),
            "toward_camera_rate":_rate(sum(bool(row.get(f"{prefix}_toward_camera")) for row in frames),complete),
            "direction_quality_rate":_rate(sum(bool(row.get(f"{prefix}_quality_valid")) for row in frames),complete),
            "valid_z0_intersection_rate":_rate(intersections,complete),
            "in_target_range_rate":_rate(in_range,intersections),
            "region_accuracy":region_accuracy(trials,prefix),
            **error_stats(row.get(f"{prefix}_error_mm") for row in trials),
            "direction_jitter_deg_mean":float(np.mean(values)) if
                (values:=_finite(trials,f"{prefix}_direction_jitter_deg")) else None,
            "origin_jitter_m_mean":float(np.mean(values)) if
                (values:=_finite(trials,f"{prefix}_origin_jitter_m")) else None,
            "hit_jitter_mm_mean":float(np.mean(values)) if
                (values:=_finite(trials,f"{prefix}_hit_jitter_mm")) else None,
            "predicted_trials":len(predicted)}


def _fallback_summary(trials,frames):
    detected=sum(bool(row.get("hand_detected")) for row in frames)
    selected_frames=[row for row in frames if row.get("selected_candidate")!="INVALID"]
    predicted=[row for row in trials if row.get("selected_region")]
    return {"complete_ray_valid_rate":_rate(len(selected_frames),detected),
            "valid_z0_intersection_rate":_rate(len(selected_frames),detected),
            "in_target_range_rate":1.0 if selected_frames else None,
            "region_accuracy":_rate(sum(bool(row.get("selected_region_correct")) for row in predicted),len(predicted)),
            **error_stats(row.get("selected_error_mm") for row in trials),
            "hit_jitter_mm_mean":float(np.mean(values)) if
                (values:=_finite(trials,"selected_hit_jitter_mm")) else None,
            "predicted_trials":len(predicted),
            "selected_frame_distribution":dict(Counter(row["selected_candidate"] for row in selected_frames))}


def _group_summary(trials,field,prefix):
    groups=defaultdict(list)
    for row in trials: groups[str(row.get(field))].append(row)
    return {name:{"trials":len(items),"region_accuracy":region_accuracy(items,prefix),
                  **error_stats(row.get(f"{prefix}_error_mm") for row in items)}
            for name,items in sorted(groups.items())}


def build_phase2b_summary(trials,frames,metadata):
    detected=sum(bool(row.get("hand_detected")) for row in frames)
    summary={"session_id":metadata["session_id"],"protocol":metadata["protocol"],
             "total_trials":len(trials),"captured_frames":len(frames),
             "intrinsics_mode":metadata["intrinsics_mode"],
             "calibration_id":metadata["calibration_id"],
             "excluded_sessions":metadata["excluded_sessions"],
             "hand_detection_rate":_rate(detected,len(frames)),
             "pointing_recognition_rate":_rate(sum(bool(row.get("pointing_state")) for row in frames),detected),
             "pnp_valid_rate":_rate(sum(bool(row.get("pnp_valid")) for row in frames),detected),
             "anchor_v2_valid_rate":_rate(sum(bool(row.get("anchor_v2_valid")) for row in frames),detected),
             "candidates":{prefix:_candidate_summary(trials,frames,prefix) for prefix in PREFIXES},
             "quality_fallback":_fallback_summary(trials,frames),
             "paired_B_C":paired_candidate_metrics(trials),
             "by_target":{prefix:_group_summary(trials,"gt_target",prefix) for prefix in ("B","C")},
             "by_hand_position":{prefix:_group_summary(trials,"hand_position",prefix) for prefix in ("B","C")},
             "by_distance_cm":{prefix:_group_summary(trials,"distance_cm",prefix) for prefix in ("B","C")}}
    intersection_ok=(summary["candidates"]["B"]["valid_z0_intersection_rate"] is not None and
              summary["candidates"]["C"]["valid_z0_intersection_rate"] is not None and
              summary["candidates"]["B"]["valid_z0_intersection_rate"]>=.5 and
              summary["candidates"]["C"]["valid_z0_intersection_rate"]>=.5)
    usable_ok=(intersection_ok and
               summary["candidates"]["B"]["in_target_range_rate"] is not None and
               summary["candidates"]["C"]["in_target_range_rate"] is not None and
               summary["candidates"]["B"]["in_target_range_rate"]>=.5 and
               summary["candidates"]["C"]["in_target_range_rate"]>=.5)
    if metadata["protocol"]=="smoke":
        summary["status"]=("PHASE_2B_LIMITED" if intersection_ok else "PHASE_2B_BLOCKED")
        summary["intersection_gate_passed"]=intersection_ok
        summary["smoke_gate_passed"]=usable_ok
    else:
        summary["status"]="PHASE_2B_LIMITED"
        summary["intersection_gate_passed"]=None
        summary["smoke_gate_passed"]=None
    failures={prefix:dict(Counter(str(row.get(f"{prefix}_failure_reason") or "UNKNOWN")
                                  for row in trials)) for prefix in PREFIXES}
    summary["failure_distribution"]=failures
    return summary


class Phase2BRecorder:
    def __init__(self,output_root,protocol,excluded_sessions):
        root=output_root if output_root.is_absolute() else PROJECT_ROOT/output_root
        self.session_id=datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        self.session_dir=safe_mkdir(root/self.session_id)
        self.screenshot_dir=safe_mkdir(self.session_dir/"screenshots")
        self.protocol=protocol; self.excluded_sessions=list(excluded_sessions)
        self.frames=[]; self.trials=[]; self.metadata={}

    def set_metadata(self,**values): self.metadata.update(values)

    def save_trial(self,rows,context,targets,minimum_valid_frames):
        trial=finalize_trial(rows,context,targets,minimum_valid_frames)
        self.frames.extend(rows); self.trials.append(trial)
        return trial

    def close(self):
        if self.metadata.get("closed"): return
        self.metadata.update({"closed":True,"session_id":self.session_id,"protocol":self.protocol,
                              "excluded_sessions":self.excluded_sessions})
        self._write_csv(self.session_dir/"frames.csv",self.frames)
        self._write_csv(self.session_dir/"trials.csv",self.trials)
        summary=build_phase2b_summary(self.trials,self.frames,self.metadata)
        with safe_open(self.session_dir/"summary.json","w",encoding="utf-8") as handle:
            json.dump(summary,handle,indent=2,ensure_ascii=False)
        self._write_candidate_artifacts(summary)
        self._write_summary(summary)
        self._write_failures(summary)
        return summary

    @staticmethod
    def _write_csv(path,rows):
        fields=list(rows[0]) if rows else ["session_id","trial_id"]
        with safe_open(path,"w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
            for row in rows: writer.writerow({key:_encoded(row.get(key)) for key in fields})

    def _write_candidate_artifacts(self,summary):
        comparisons=["# B/C Paired Candidate Comparison","",
                     f"- Paired trials: {summary['paired_B_C']['paired_trials']}",
                     f"- B wins: {summary['paired_B_C']['B_win_count']}",
                     f"- C wins: {summary['paired_B_C']['C_win_count']}",
                     f"- Ties: {summary['paired_B_C']['tie_count']}",
                     f"- Median C-B error: {summary['paired_B_C']['median_C_minus_B_mm']} mm",
                     f"- P90 C-B error: {summary['paired_B_C']['p90_C_minus_B_mm']} mm","",
                     "Selection is quality-only; Ground Truth is never read by fallback logic."]
        with safe_open(self.session_dir/"candidate_comparison.md","w",encoding="utf-8") as handle:
            handle.write("\n".join(comparisons))
        for prefix in ("B","C"):
            matrix={gt:{pred:0 for pred in REGION_ORDER} for gt in REGION_ORDER}
            for row in self.trials:
                gt,pred=row.get("gt_target"),row.get(f"{prefix}_region")
                if gt in matrix and pred in matrix[gt]: matrix[gt][pred]+=1
            path=self.session_dir/f"region_confusion_matrix_{prefix}.csv"
            with safe_open(path,"w",newline="",encoding="utf-8") as handle:
                writer=csv.writer(handle); writer.writerow(["GT/PRED",*REGION_ORDER])
                for gt in REGION_ORDER: writer.writerow([gt,*[matrix[gt][pred] for pred in REGION_ORDER]])
            self._plot_matrix(matrix,prefix)
            self._plot_errors(prefix)

    def _plot_matrix(self,matrix,prefix):
        os.environ.setdefault("MPLCONFIGDIR",str(self.session_dir/"matplotlib_cache"))
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        values=np.asarray([[matrix[gt][pred] for pred in REGION_ORDER] for gt in REGION_ORDER])
        figure=Figure(figsize=(8,7),dpi=120); FigureCanvasAgg(figure); axis=figure.subplots()
        axis.imshow(values,cmap="Blues",vmin=0)
        axis.set_xticks(range(9),REGION_ORDER,rotation=45,ha="right",fontsize=7)
        axis.set_yticks(range(9),REGION_ORDER,fontsize=7); axis.set_title(f"Candidate {prefix}")
        for row in range(9):
            for col in range(9): axis.text(col,row,str(values[row,col]),ha="center",va="center")
        figure.tight_layout(); buffer=BytesIO(); figure.savefig(buffer,format="png")
        with safe_open(self.session_dir/f"region_confusion_matrix_{prefix}.png","wb") as handle:
            handle.write(buffer.getvalue())

    def _plot_errors(self,prefix):
        os.environ.setdefault("MPLCONFIGDIR",str(self.session_dir/"matplotlib_cache"))
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        values=_finite(self.trials,f"{prefix}_error_mm")
        figure=Figure(figsize=(7,4),dpi=120); FigureCanvasAgg(figure); axis=figure.subplots()
        if values: axis.bar(range(1,len(values)+1),values,color="#d62728" if prefix=="C" else "#1f77b4")
        axis.set_xlabel("Valid paired/trial prediction"); axis.set_ylabel("Landing error (mm)")
        axis.set_title(f"Candidate {prefix} landing error"); figure.tight_layout()
        buffer=BytesIO(); figure.savefig(buffer,format="png")
        with safe_open(self.session_dir/f"landing_error_{prefix}.png","wb") as handle:
            handle.write(buffer.getvalue())

    def _write_summary(self,summary):
        def value(item): return "N/A" if item is None else f"{item:.3f}" if isinstance(item,float) else str(item)
        lines=["# Phase 2B Z=0 3D Pointing Report","","## 1. Purpose","",
               "Validate whether calibrated Camera-space B/C rays produce usable Z=0 A4 landing points.","",
               "## 2. System Architecture","",
               "MediaPipe world hand pose -> PnP rotation -> V2 robust palm anchor -> TIP Camera origin -> parallel B/C rays -> Z=0 intersection -> eval-coordinate A4 landing point.","",
               "## 3. Camera Calibration","",
               f"- Intrinsics: {summary['intrinsics_mode']}",f"- Calibration ID: {summary['calibration_id']}",
               "- Calibration RMS: 1.6118 px","- Formal resolution: 1280 x 720","",
               "## 4. Test Protocol","",f"- Protocol: {summary['protocol']}",
               f"- Trials: {summary['total_trials']}",f"- Frames: {summary['captured_frames']}",
               f"- Excluded invalid sessions: {', '.join(summary['excluded_sessions'])}","",
               "## 5. Baseline A","",f"```json\n{json.dumps(summary['candidates']['A'],indent=2,ensure_ascii=False)}\n```","",
               "## 6. Candidate B","",f"```json\n{json.dumps(summary['candidates']['B'],indent=2,ensure_ascii=False)}\n```","",
               "## 7. Candidate C","",f"```json\n{json.dumps(summary['candidates']['C'],indent=2,ensure_ascii=False)}\n```","",
               "## 8. B/C Paired Comparison","",f"```json\n{json.dumps(summary['paired_B_C'],indent=2,ensure_ascii=False)}\n```","",
               "### C-to-B quality fallback","",f"```json\n{json.dumps(summary['quality_fallback'],indent=2,ensure_ascii=False)}\n```","",
               "## 9. Z=0 Intersection Results","",
               f"- B valid intersection rate: {value(summary['candidates']['B']['valid_z0_intersection_rate'])}",
               f"- C valid intersection rate: {value(summary['candidates']['C']['valid_z0_intersection_rate'])}","",
               "## 10. Region Accuracy","",
               f"- B: {value(summary['candidates']['B']['region_accuracy'])}",
               f"- C: {value(summary['candidates']['C']['region_accuracy'])}","",
               "## 11. Landing Error","","| Candidate | Mean mm | Median mm | P90 mm | Max mm |",
               "|---|---:|---:|---:|---:|",
               *[f"| {p} | {value(summary['candidates'][p]['mean_mm'])} | {value(summary['candidates'][p]['median_mm'])} | {value(summary['candidates'][p]['p90_mm'])} | {value(summary['candidates'][p]['max_mm'])} |" for p in ('B','C')],"",
               "## 12. Hand-position Robustness","",f"```json\n{json.dumps(summary['by_hand_position'],indent=2,ensure_ascii=False)}\n```","",
               "## 13. Distance Robustness","",f"```json\n{json.dumps(summary['by_distance_cm'],indent=2,ensure_ascii=False)}\n```","",
               "## 14. Failure Distribution","",f"```json\n{json.dumps(summary['failure_distribution'],indent=2,ensure_ascii=False)}\n```","",
               "## 15. Current Limitations","",
               "This is current calibrated-camera experimental landing accuracy, not industrial millimetric accuracy. Scene metric depth remains a global-distance estimate and PnP/calibration residuals remain error sources.","",
               "## 16. Delivery Recommendation","",f"**{summary['status']}**","",
               "B/C remain independently reported. Quality fallback never reads GT. Full evaluation is allowed only after the Smoke gate passes."]
        with safe_open(self.session_dir/"summary.md","w",encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def _write_failures(self,summary):
        lines=["# Phase 2B Failure Analysis","",f"Session: {self.session_id}",""]
        for prefix,items in summary["failure_distribution"].items():
            lines.extend([f"## Candidate {prefix}","","| Reason | Trials |","|---|---:|",
                          *[f"| {reason} | {count} |" for reason,count in sorted(items.items())],""])
        lines.append("No failed frame or Trial is deleted. OUT_OF_TARGET_RANGE is never clamped to a region.")
        with safe_open(self.session_dir/"failure_analysis.md","w",encoding="utf-8") as handle:
            handle.write("\n".join(lines))
