from __future__ import annotations

import os
from pathlib import Path

from src.safety.path_guard import PROJECT_ROOT, safe_mkdir


def isolate_runtime() -> None:
    """Redirect common caches and temporary writes into the project."""
    cache = safe_mkdir(PROJECT_ROOT / "cache")
    mappings = {
        "TORCH_HOME": PROJECT_ROOT / "models" / "torch",
        "HF_HOME": PROJECT_ROOT / "models" / "huggingface",
        "HUGGINGFACE_HUB_CACHE": PROJECT_ROOT / "models" / "huggingface" / "hub",
        "XDG_CACHE_HOME": cache,
        "PIP_CACHE_DIR": cache / "pip",
        "MPLCONFIGDIR": cache / "matplotlib",
        "TEMP": cache / "temp",
        "TMP": cache / "temp",
        "MEDIAPIPE_HOME": PROJECT_ROOT / "models" / "mediapipe",
    }
    for key, value in mappings.items():
        path = safe_mkdir(value)
        os.environ[key] = str(path)
    # Some restricted Windows shells omit WINDIR; matplotlib (imported by
    # MediaPipe) needs it only to discover system fonts. This is process-local.
    os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", r"C:\Windows"))


def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
