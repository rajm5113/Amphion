"""Unified scoring API over the trained models (hybrid backend).

    from amphion import score_activity, score_toxicity
    score_activity("GIGKFLHSAKKFGKAFVGEIMNS")
    # {'amp_prob': 0.99, 'pred_log_mic': 0.83, 'pred_mic_uM': 6.7}

Hybrid (set by the ESM-2 benchmark — each model where it wins):
  - activity  -> ESM-2 embeddings if available (AUC 0.97), else physicochemical baseline
  - toxicity  -> ESM-2 embeddings if available, else physicochemical baseline
  - MIC       -> compact physicochemical model (beat ESM on potency regression)

Models load lazily on first use and are cached, so ``import amphion`` never
requires the artifacts (or ESM-2) to exist.
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
            f"(see the README Quickstart, or run the ESM-2 notebook for the *_esm models)."
        )
    # The ESM heads are trained on Kaggle (scikit-learn 1.6.x) and load here under a
    # newer sklearn -> a benign InconsistentVersionWarning. Predictions are verified
    # correct; suppress only this specific, documented warning to keep output clean.
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


def _load_preferred(esm_name: str, base_name: str):
    """Prefer the ESM-2 model if present, else fall back to the physicochemical baseline."""
    cfg = get_config()
    if (cfg.resolve_path("models") / esm_name).exists():
        return _load(esm_name)
    return _load(base_name)


def _features(artifact, seq: str):
    """Featurize one sequence with the backend the artifact was trained on."""
    if artifact.get("features") == "esm":
        from .features.esm_embed import embed_sequences

        return embed_sequences([seq], artifact["esm"])
    return featurize(seq).reshape(1, -1)


def _validate(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s or any(ch not in CANONICAL_SET for ch in s):
        raise ValueError(f"sequence must use only the 20 standard amino acids: {seq!r}")
    return s


def score_activity(seq: str) -> dict:
    """Return {amp_prob, pred_log_mic, pred_mic_uM} for one peptide."""
    s = _validate(seq)
    act = _load_preferred("activity_clf_esm.joblib", "activity_clf.joblib")
    mic = _load("mic_reg.joblib")  # hybrid: MIC always uses the compact physicochemical model
    amp_prob = float(act["model"].predict_proba(_features(act, s))[0, 1])
    pred_log_mic = float(mic["model"].predict(_features(mic, s))[0])
    return {
        "amp_prob": amp_prob,
        "pred_log_mic": pred_log_mic,
        "pred_mic_uM": float(10.0 ** pred_log_mic),
    }


def score_toxicity(seq: str) -> dict:
    """Return {hemolytic_prob} for one peptide (probability it harms human RBCs)."""
    s = _validate(seq)
    tox = _load_preferred("tox_clf_esm.joblib", "tox_clf.joblib")
    return {"hemolytic_prob": float(tox["model"].predict_proba(_features(tox, s))[0, 1])}


def active_backends() -> dict:
    """Which backend each model is using right now ('esm' or 'baseline') — for reports/UI."""
    cfg = get_config()
    m = cfg.resolve_path("models")
    return {
        "activity": "esm" if (m / "activity_clf_esm.joblib").exists() else "baseline",
        "toxicity": "esm" if (m / "tox_clf_esm.joblib").exists() else "baseline",
        "mic": "baseline",
    }
