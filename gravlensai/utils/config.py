"""YAML config loading helpers."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_yaml_section(config_path: str, section: str) -> dict[str, Any]:
    """
    Load one top-level section from a YAML config file.

    Returns an empty dict if the file/section is missing.
    """
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    value = data.get(section, {})
    return value if isinstance(value, dict) else {}