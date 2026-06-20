"""Unified scoring API over the trained models.

    from amphion import score_activity
    score_activity("GIGKFLHSAKKFGKAFVGEIMNS")
    # {'amp_prob': 0.99, 'pred_log_mic': 0.83, 'pred_mic_uM': 6.7}

Models are loaded lazily on first use and cached, so ``import amphion`` never
requires the artifacts to exist.
"""

from __future__ import annotations

from functools import lru_cache

import joblib

from .config import get_config
from .features.featurize import featurize
from .utils import CANONICAL_SET


@lru_cache(maxsize=None)
def _load(name: str):
    cfg = get_config()
    path = cfg.resolve_path("models") / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — train it first "
            f"(python -m amphion.models.{name.replace('.joblib', '')})"
        )
    return joblib.load(path)


def _validate(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s or any(ch not in CANONICAL_SET for ch in s):
        raise ValueError(f"sequence must use only the 20 standard amino acids: {seq!r}")
    return s


def score_activity(seq: str) -> dict:
    """Return {amp_prob, pred_log_mic, pred_mic_uM} for one peptide."""
    s = _validate(seq)
    x = featurize(s).reshape(1, -1)
    clf = _load("activity_clf.joblib")
    reg = _load("mic_reg.joblib")
    amp_prob = float(clf["model"].predict_proba(x)[0, 1])
    pred_log_mic = float(reg["model"].predict(x)[0])
    return {
        "amp_prob": amp_prob,
        "pred_log_mic": pred_log_mic,
        "pred_mic_uM": float(10.0 ** pred_log_mic),
    }
