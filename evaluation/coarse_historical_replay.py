from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path


REQUIRED_EXACT_FIELDS = {
    "image_landmarks_px", "world_landmarks_m", "tip_pixel_raw"
}


def csv_header(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle), []))


def audit_session(session_path, csv_name):
    path = Path(session_path)
    data_path = path / csv_name
    markers = sorted(item.name for item in path.glob("INVALID*.txt"))
    if not data_path.exists():
        return {"session": path.name, "usable": False, "reason": "DATA_FILE_MISSING",
                "markers": markers, "missing_fields": sorted(REQUIRED_EXACT_FIELDS)}
    header = csv_header(data_path)
    missing = sorted(REQUIRED_EXACT_FIELDS - header)
    return {
        "session": path.name,
        "data_file": str(data_path.relative_to(PROJECT_ROOT)),
        "markers": markers,
        "available_field_count": len(header),
        "missing_fields": missing,
        "usable": not missing and not markers,
        "reason": "EXACT_REPLAY_READY" if not missing and not markers else (
            "EXCLUDED_BY_MARKER" if markers else "MISSING_RAW_FIELDS"
        ),
    }


def build_audit():
    config = load_yaml(PROJECT_ROOT / "configs/mentor_coarse_pointing.yaml")
    historical = config["historical"]
    sessions = [
        audit_session(PROJECT_ROOT / historical["phase2a5_clean"], "samples.csv"),
        audit_session(PROJECT_ROOT / historical["phase2b_official_smoke"], "frames.csv"),
    ]
    sessions.extend(audit_session(PROJECT_ROOT / item, "frames.csv")
                    for item in historical["failure_diagnostics"])
    incomplete = PROJECT_ROOT / "reports/3d_pointing_phase2b/session_20260826_205359_827058"
    sessions.append(audit_session(incomplete, "frames.csv"))
    return {
        "mode": "CAMERA_CENTERED_COARSE_REGION",
        "exact_replay_possible": all(item["usable"] for item in sessions[:2]),
        "sessions": sessions,
        "critical_missing_fields": sorted(REQUIRED_EXACT_FIELDS),
        "why_required": {
            "image_landmarks_px": "Required to recompute current/RANSAC/palm PnP and 2D projection agreement.",
            "world_landmarks_m": "Required to recompute B/C and all orientation hypotheses.",
            "tip_pixel_raw": "Required by the calibrated depth-free Hx/Hy formula.",
        },
        "historical_ray_proxy_allowed_for_acceptance": False,
        "new_human_test_if_needed": "At most one CENTER/LEFT/RIGHT/UP/DOWN trial (5 total).",
    }


def main():
    output = build_audit()
    path = assert_safe_path(PROJECT_ROOT / "reports/historical_coarse_replay_audit.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
