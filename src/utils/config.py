from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(
    os.environ.get(
        "SAM3D_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()


def resolve_project_path(path: str | Path | None) -> Path:
    if path is None:
        return REPO_ROOT

    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj

    return (REPO_ROOT / path_obj).resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"

    config_path = resolve_project_path(config_path)

    if not config_path.is_file():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a mapping/dict, got {type(data).__name__}")

    return data