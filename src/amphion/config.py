"""Load and access ``config.yaml``.

Use :func:`get_config` everywhere instead of hard-coding thresholds or paths.
Nested values are reachable by attribute *or* item access::

    cfg = get_config()
    cfg.seed                       # 42
    cfg.activity.active_mic_uM_max # 32
    cfg["length"]["max"]           # 50
    cfg.resolve_path("processed")  # absolute Path under the repo root
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


class Config(dict):
    """A ``dict`` that also supports attribute access (recursively)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - thin wrapper
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    # -- convenience -------------------------------------------------------
    def resolve_path(self, key: str) -> Path:
        """Resolve a key under ``paths:`` to an absolute path under the repo root."""
        rel = self["paths"][key]
        return (repo_root() / rel).resolve()


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def repo_root() -> Path:
    """Repository root: two levels up from this file (src/amphion/config.py)."""
    return Path(__file__).resolve().parents[2]


def find_config_file(start: str | os.PathLike | None = None) -> Path:
    """Find ``config.yaml`` by walking up from ``start`` (default cwd), then the repo root."""
    names = ("config.yaml", "config.yml")
    start_dir = Path(start or os.getcwd()).resolve()
    search_dirs = [start_dir, *start_dir.parents, repo_root()]
    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"config.yaml not found (searched from {start_dir} up to filesystem root, "
        f"and repo root {repo_root()})"
    )


@lru_cache(maxsize=None)
def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config from ``path`` (or auto-discovered ``config.yaml``). Cached."""
    cfg_path = Path(path) if path else find_config_file()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _to_config(raw)
    cfg["_path"] = str(cfg_path)
    return cfg


def get_config(path: str | os.PathLike | None = None) -> Config:
    """Alias for :func:`load_config` — the preferred public entry point."""
    return load_config(path)
