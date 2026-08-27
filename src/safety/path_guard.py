from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assert_safe_path(path: str | Path) -> Path:
    """Resolve a prospective write target and reject paths outside the project."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Refusing write outside project root: {resolved}") from exc
    return resolved


def safe_mkdir(path: str | Path) -> Path:
    target = assert_safe_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_open(path: str | Path, mode: str = "r", **kwargs):
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        target = assert_safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = Path(path).resolve(strict=False)
    return target.open(mode, **kwargs)

