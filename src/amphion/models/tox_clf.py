"""Toxicity classifier: predict whether a peptide is hemolytic (harms RBCs).

Trains on the HemoPI2-derived hemolysis dataset (label = HC50 <= threshold).
Imbalance-aware: class weights + PR-AUC (average precision) reported ALONGSIDE
ROC-AUC. Cluster-aware CV (StratifiedGroupKFold on homology cluster_id).
Selects the best by cluster-aware PR-AUC, isotonic-calibrates, fits on all data,
saves models/tox_clf.joblib, and writes reports/toxicity_model_card.md.
"""

from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from ..config import get_config
from ..features.featurize import FEATURE_NAMES, featurize_many
from ..utils import ensure_dir, get_logger, set_seed

log = get_logger("amphion.models.tox_clf")

# Sanity probes: Melittin is strongly hemolytic; the others much less so.
PROBES = {
    "Melittin (hemolytic)": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "Magainin-2 (low hemolysis)": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Polar control": "SEEGDTAATGGDSTGAESDTAAGSE",
}


def _load(cfg):
    df = pd.read_parquet(cfg.resolve_path("processed") / "hemolysis.parquet")
    X = featurize_many(df.sequence.tolist())
    y = df.hemolytic.to_numpy()
    groups = df.cluster_id.to_numpy()
    return X, y, groups


def _models(seed):
    return {
        "LogReg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced"
        ),
        "HistGB": HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"),
    }


def _metrics(y, p):
    pr = (p >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Acc": accuracy_score(y, pr),
        "F1": f1_score(y, pr),
        "MCC": matthews_corrcoef(y, pr),
    }


def train(cfg=None) -> dict:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    X, y, groups = _load(cfg)
    base_rate = float(y.mean())
    log.info("hemolysis set: X=%s, hemolytic=%d (%.1f%%), clusters=%d",
             X.shape, int(y.sum()), 100 * base_rate, len(set(groups)))

    cv = StratifiedGroupKFold(5)
    results = {}
    for name, model in _models(cfg.seed).items():
        p = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        results[name] = _metrics(y, p)
        log.info("%-12s ROC-AUC=%.3f PR-AUC=%.3f MCC=%.3f",
                 name, results[name]["ROC_AUC"], results[name]["PR_AUC"], results[name]["MCC"])

    best = max(results, key=lambda n: results[n]["PR_AUC"])
    log.info("best by cluster-aware PR-AUC: %s", best)

    calibrated = CalibratedClassifierCV(_models(cfg.seed)[best], method="isotonic", cv=5)
    calibrated.fit(X, y)

    probes = {n: float(calibrated.predict_proba(featurize_many([s]))[0, 1]) for n, s in PROBES.items()}

    models_dir = ensure_dir(cfg.resolve_path("models"))
    joblib.dump(
        {"kind": "tox_clf", "model": calibrated, "base_model": best, "features": FEATURE_NAMES,
         "minlen": cfg.length.min, "maxlen": cfg.length.max, "metrics": results[best],
         "base_rate": base_rate},
        models_dir / "tox_clf.joblib",
    )
    log.info("saved %s", models_dir / "tox_clf.joblib")

    _write_card(cfg, results, best, base_rate, probes, X.shape)
    return {"best": best, "results": results, "probes": probes}


def _write_card(cfg, results, best, base_rate, probes, shape):
    rows = "\n".join(
        f"| {n} | {m['ROC_AUC']:.3f} | {m['PR_AUC']:.3f} | {m['F1']:.3f} | {m['MCC']:.3f} |"
        for n, m in results.items()
    )
    probe_rows = "\n".join(f"| {n} | {p:.2f} |" for n, p in probes.items())
    thr = cfg.toxicity.hemolytic_hc50_uM_max
    card = f"""# Model card — Toxicity classifier (hemolysis)

**Task:** predict whether a peptide is hemolytic (ruptures human red blood cells).
**Data:** HemoPI2 (Raghava lab), {shape[0]:,} peptides, {shape[1]} features.
**Label:** hemolytic if **HC50 ≤ {thr} µM** (lower HC50 = lyses RBCs at lower dose
= more toxic). This threshold reproduces HemoPI2's own binary labels 100%.
**Base rate:** {100*base_rate:.1f}% hemolytic (mild imbalance → PR-AUC reported).
**Selected model:** {best} (best cluster-aware PR-AUC), isotonic-calibrated.

## Cross-validated performance (cluster-aware, StratifiedGroupKFold)
| Model | ROC-AUC | PR-AUC | F1 | MCC |
|---|---:|---:|---:|---:|
{rows}

PR-AUC is reported alongside ROC-AUC because the classes are imbalanced; a random
baseline PR-AUC equals the base rate ({base_rate:.2f}).

## Sanity probes (calibrated hemolytic probability)
| Peptide | hemolytic_prob |
|---|---:|
{probe_rows}

Melittin (a classic hemolytic peptide) should score high; benign controls low.

## Use in the pipeline
`score_toxicity(seq) → {{hemolytic_prob}}`. The assembly loop (Phase 6) drops any
candidate with `hemolytic_prob > toxicity.hemolytic_prob_max`
(default {cfg.toxicity.hemolytic_prob_max}; lower it to be stricter).

> Computational predictions with uncertainty, not validated safety results.
"""
    out = ensure_dir(cfg.resolve_path("reports")) / "toxicity_model_card.md"
    out.write_text(card, encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    train()
