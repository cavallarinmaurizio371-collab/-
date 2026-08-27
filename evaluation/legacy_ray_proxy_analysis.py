from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experimental_3d_pointing.coarse import map_coarse_region
from src.safety.path_guard import assert_safe_path


def _vector(value):
    try:
        result = np.asarray(ast.literal_eval(value), dtype=float)
        return result if result.shape == (3,) and np.all(np.isfinite(result)) else None
    except (ValueError, SyntaxError, TypeError):
        return None


def _truth(value):
    return str(value).strip().lower() == "true"


def _legacy_origin_proxy(origin, direction):
    """Diagnostic only: old reconstructed origin, not the required TIP-pixel hit."""
    if origin is None or direction is None or origin[2] <= 0 or direction[2] >= -1e-6:
        return None
    camera = np.asarray([origin[0] / origin[2] - direction[0] / direction[2],
                         origin[1] / origin[2] - direction[1] / direction[2]])
    eval_xy = -camera
    return float(np.degrees(np.arctan(eval_xy[0]))), float(np.degrees(np.arctan(eval_xy[1])))


def analyze_phase2b(path):
    grouped = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["trial_id"]].append(row)
    trials = []
    for trial_id, rows in grouped.items():
        item = {"trial_id": trial_id, "target": rows[0]["gt_target"]}
        per_candidate = {}
        for prefix in ("B", "C"):
            values = []
            for row in rows:
                if not _truth(row["pointing_state"]):
                    continue
                hit = _legacy_origin_proxy(_vector(row[f"{prefix}_origin"]),
                                           _vector(row[f"{prefix}_direction"]))
                if hit is not None:
                    values.append(hit)
            if values:
                median = np.median(np.asarray(values), axis=0)
                region = map_coarse_region(median[0], median[1])
                per_candidate[prefix] = {"valid": True, "yaw_deg": float(median[0]),
                    "pitch_deg": float(median[1]), "region": region,
                    "valid_frames": len(values)}
            else:
                per_candidate[prefix] = {"valid": False, "region": None,
                                         "valid_frames": 0}
        item.update(per_candidate); trials.append(item)
    summary = {}
    for prefix in ("B", "C"):
        valid = [trial for trial in trials if trial[prefix]["valid"]]
        correct = [trial for trial in valid if trial[prefix]["region"] == trial["target"]]
        summary[prefix] = {"valid_trials": len(valid), "total_trials": len(trials),
                           "accuracy": len(correct) / len(valid) if valid else None}
    return {"trials": trials, "summary": summary}


def analyze_phase2a5(path):
    grouped = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["trial_id"]].append(row)
    output = {"B": [], "C": []}
    names = {"B": "b_relative", "C": "c_axis_fit"}
    for prefix, base in names.items():
        for trial_id, rows in grouped.items():
            values = [_legacy_origin_proxy(_vector(row[f"{base}_origin"]),
                                           _vector(row[f"{base}_smoothed_direction"]))
                      for row in rows if _truth(row["pointing"])]
            values = [value for value in values if value is not None]
            if values:
                median = np.median(np.asarray(values), axis=0)
                output[prefix].append({"trial_id": trial_id, "pose": rows[0]["pose"],
                    "region": map_coarse_region(median[0], median[1]),
                    "yaw_deg": float(median[0]), "pitch_deg": float(median[1])})
    return {prefix: {"center_count": sum(item["region"] == "CENTER" for item in values),
                     "trial_count": len(values), "trials": values}
            for prefix, values in output.items()}


def main():
    official = PROJECT_ROOT / "reports/3d_pointing_phase2b/session_20260826_193216_892285/frames.csv"
    reference = PROJECT_ROOT / "reports/phase2a5_sessions/session_20260826_191039_450446/samples.csv"
    report = {
        "warning": "LEGACY_ORIGIN_PROXY_ONLY_NOT_DEPTH_FREE_NOT_FOR_DELIVERY_ACCEPTANCE",
        "official_smoke": analyze_phase2b(official),
        "phase2a5_center_reference": analyze_phase2a5(reference),
    }
    path = assert_safe_path(PROJECT_ROOT / "reports/legacy_ray_coarse_proxy.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    print(json.dumps(report["official_smoke"]["summary"], indent=2))
    print(json.dumps({key: {"center_count": value["center_count"],
                           "trial_count": value["trial_count"]}
                      for key, value in report["phase2a5_center_reference"].items()}, indent=2))


if __name__ == "__main__":
    main()
