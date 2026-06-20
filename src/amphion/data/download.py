"""Fetch the public source datasets into ``data/raw/`` (git-ignored).

Sources (URLs live in ``config.yaml`` ``data_sources:``):
  - GRAMPA            : consolidated AMP MIC table (positives + MIC labels)
  - AMPlify non-AMP   : UniProt-derived non-antimicrobial proteins (negatives)

Hemolytik/DRAMP toxicity sources are added in Phase 3.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from ..config import get_config
from ..utils import ensure_dir, get_logger

log = get_logger("amphion.data.download")

_HEADERS = {"User-Agent": "amphion/0.1 (research; +https://github.com/)"}


def download_file(url: str, dest: str | Path, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (use ``force`` to refetch)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("exists: %s (%d bytes)", dest.name, dest.stat().st_size)
        return dest
    ensure_dir(dest.parent)
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    log.info("saved %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def download_all(cfg=None, force: bool = False) -> dict[str, Path]:
    """Download every Phase-1 (activity) source; return a name -> path map."""
    cfg = cfg or get_config()
    raw = cfg.resolve_path("raw")
    src = cfg.data_sources
    return {
        "grampa": download_file(src.grampa, raw / "grampa.csv", force),
        "neg_train": download_file(src.neg_train, raw / "AMPlify_non_AMP_train_balanced.fa", force),
        "neg_test": download_file(src.neg_test, raw / "AMPlify_non_AMP_test_balanced.fa", force),
    }


def download_toxicity(cfg=None, force: bool = False) -> dict[str, Path]:
    """Download the Phase-3 hemolysis sources (HemoPI2: SEQUENCE, HC50 uM, label)."""
    cfg = cfg or get_config()
    raw = cfg.resolve_path("raw")
    src = cfg.data_sources
    return {
        "hemopi2_crossval": download_file(src.hemopi2_crossval, raw / "hemopi2_cross_val.csv", force),
        "hemopi2_independent": download_file(src.hemopi2_independent, raw / "hemopi2_independent.csv", force),
    }


if __name__ == "__main__":
    for name, path in {**download_all(), **download_toxicity()}.items():
        print(f"{name:20s} -> {path}")
