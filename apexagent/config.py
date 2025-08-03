"""YAML config loading with dotted-key CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load the default config, merge an optional user config on top, then apply overrides.

    Overrides look like ``ppo.learning_rate=1e-4`` or ``env.tracks=[oval,kidney]``; values are
    parsed with YAML so numbers, booleans and lists work naturally.
    """
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key=value")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise KeyError(f"cannot set '{key}': '{p}' is not a section")
    node[parts[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
