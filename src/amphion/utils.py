"""Shared utilities: deterministic seeding, logging, small IO + hashing helpers.

All randomness in Amphion should be routed through :func:`set_seed` so results
are reproducible (determinism matters for a portfolio).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# The 20 standard amino acids — the only alphabet Amphion accepts.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_SET = frozenset(CANONICAL_AA)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> int:
    """Seed Python, NumPy, and (if installed) PyTorch. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is only present in GPU phases
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass
    return seed


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "amphion", level: int = logging.INFO) -> logging.Logger:
    """A console logger that won't add duplicate handlers on repeated calls."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# FASTA
# --------------------------------------------------------------------------- #
def read_fasta(path: str | os.PathLike) -> list[str]:
    """Return the sequences (uppercased, concatenated per record) from a FASTA file."""
    seqs: list[str] = []
    cur: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    return seqs


# --------------------------------------------------------------------------- #
# Hashing / manifests (reproducibility)
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_sequences(seqs: Iterable[str]) -> str:
    """Order-independent content hash of a set of sequences (for manifests)."""
    joined = "\n".join(sorted(seqs))
    return sha256_text(joined)


def write_json(path: str | os.PathLike, obj) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return p
