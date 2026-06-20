"""Activity classifier: predict AMP vs non-AMP from sequence.

Trains LogReg / RandomForest / HistGradientBoosting on the balanced,
length-matched subset. Reports BOTH:
  - random 5-fold CV (reproduces the known-good baseline, ~0.97 AUC), and
  - cluster-aware CV (StratifiedGroupKFold on homology cluster_id) — the HONEST
    number that doesn't leak near-duplicates across folds.
Selects the best by cluster-aware AUC, calibrates probabilities (isotonic),
fits on all balanced data, saves to models/activity_clf.joblib, and writes
reports/activity_model_card.md including the biological sanity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from ..config import get_config
from ..features.featurize import FEATURE_NAMES, featurize_many
from ..utils import ensure_dir, get_logger, set_seed

log = get_logger("amphion.models.activity_clf")

# Charge-related features — the biological sanity check expects these to dominate.
CHARGE_FEATURES = {"net_charge", "charge_density", "frac_pos", "K", "R"}

KNOWN = {
    "Magainin-2 (AMP)": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Melittin (AMP)": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "Random non-AMP": "SEEGDTAATGGDSTGAESDTAAGSE",
}


def _load_balanced(cfg):
    df = pd.read_parquet(cfg.resolve_path("processed") / "activity_classification.parquet")
    df = df[df.in_balanced].reset_index(drop=True)
    X = featurize_many(df.sequence.tolist())
    y = df.label.to_numpy()
    groups = df.cluster_id.to_numpy()
    return X, y, groups


def _models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }


def _metrics(y, p):
    pr = (p >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "Acc": accuracy_score(y, pr),
        "F1": f1_score(y, pr),
        "MCC": matthews_corrcoef(y, pr),
        "cm": confusion_matrix(y, pr).tolist(),
    }


def train(cfg=None) -> dict:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    X, y, groups = _load_balanced(cfg)
    log.info("balanced set: X=%s, positives=%d, groups=%d", X.shape, int(y.sum()), len(set(groups)))

    rand_cv = StratifiedKFold(5, shuffle=True, random_state=cfg.seed)
    grp_cv = StratifiedGroupKFold(5)

    results = {}
    for name, model in _models(cfg.seed).items():
        p_rand = cross_val_predict(model, X, y, cv=rand_cv, method="predict_proba")[:, 1]
        p_grp = cross_val_predict(model, X, y, cv=grp_cv, groups=groups, method="predict_proba")[:, 1]
        results[name] = {"random": _metrics(y, p_rand), "cluster_aware": _metrics(y, p_grp)}
        log.info(
            "%-12s random AUC=%.3f | cluster-aware AUC=%.3f MCC=%.3f",
            name, results[name]["random"]["AUC"],
            results[name]["cluster_aware"]["AUC"], results[name]["cluster_aware"]["MCC"],
        )

    best_name = max(results, key=lambda n: results[n]["cluster_aware"]["AUC"])
    log.info("best by cluster-aware AUC: %s", best_name)

    # calibrate the winner (isotonic) and fit on all balanced data
    base = _models(cfg.seed)[best_name]
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X, y)

    # biological sanity check: feature importances from a plain RF
    rf = RandomForestClassifier(n_estimators=400, random_state=cfg.seed, n_jobs=-1).fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])
    top = [n for n, _ in imp[:6]]
    charge_in_top = [n for n in top if n in CHARGE_FEATURES]
    sanity_pass = len(charge_in_top) >= 2
    log.info("top features: %s  | charge-related in top6: %s", top, charge_in_top)

    # known-peptide scores (calibrated)
    known_scores = {name: float(calibrated.predict_proba(featurize_many([seq]))[0, 1])
                    for name, seq in KNOWN.items()}

    # save artifact
    models_dir = ensure_dir(cfg.resolve_path("models"))
    artifact = {
        "kind": "activity_clf",
        "model": calibrated,
        "base_model": best_name,
        "features": FEATURE_NAMES,
        "minlen": cfg.length.min,
        "maxlen": cfg.length.max,
        "metrics": results[best_name],
        "feature_importances": imp,
    }
    out = models_dir / "activity_clf.joblib"
    joblib.dump(artifact, out)
    log.info("saved %s", out)

    _write_card(cfg, results, best_name, imp, charge_in_top, sanity_pass, known_scores, X.shape)
    if not sanity_pass:
        log.warning("BIOLOGY SANITY CHECK FAILED — charge features did not dominate")
    return {"best": best_name, "results": results, "sanity_pass": sanity_pass, "known": known_scores}


def _write_card(cfg, results, best, imp, charge_in_top, sanity_pass, known, shape):
    def row(name):
        r, g = results[name]["random"], results[name]["cluster_aware"]
        return (f"| {name} | {r['AUC']:.3f} | {g['AUC']:.3f} | {g['Acc']:.3f} | "
                f"{g['F1']:.3f} | {g['MCC']:.3f} |")

    top_lines = "\n".join(f"| {n} | {v:.3f} |" for n, v in imp[:12])
    known_lines = "\n".join(f"| {n} | {s:.2f} |" for n, s in known.items())
    card = f"""# Model card — Activity classifier (AMP vs non-AMP)

**Task:** predict whether a peptide is antimicrobial, from sequence alone.
**Data:** balanced, length-matched subset ({shape[0]:,} peptides, {shape[1]} features).
**Selected model:** {best} (chosen by cluster-aware AUC), isotonic-calibrated.

## Cross-validated performance
Two CV schemes. **Cluster-aware** (group by homology cluster) is the honest number;
random split is shown only to reproduce the known-good baseline.

| Model | Random AUC | Cluster-aware AUC | Acc | F1 | MCC |
|---|---:|---:|---:|---:|---:|
{row("LogReg")}
{row("RandomForest")}
{row("HistGB")}

> Random-split RF AUC ≈ 0.97 reproduces the baseline; the cluster-aware AUC is
> lower and more honest — it forbids near-duplicate sequences leaking across folds.

## Biological sanity check {'✅ PASS' if sanity_pass else '❌ FAIL'}
Top features should be charge-related (AMPs are cationic and disrupt the
negatively-charged bacterial membrane). Charge-related features in the top 6:
**{', '.join(charge_in_top) if charge_in_top else 'none'}**.

| Feature | Importance |
|---|---:|
{top_lines}

## Known-peptide check (calibrated probability)
| Peptide | amp_prob |
|---|---:|
{known_lines}

Known AMPs should score ≈ 1.0, the random control low.

## Calibration & use
Probabilities are isotonic-calibrated so `amp_prob` is a usable confidence.
Full reliability diagrams + Brier score come in Phase 7.

> These are computational predictions with uncertainty, not validated results.
"""
    out = ensure_dir(cfg.resolve_path("reports")) / "activity_model_card.md"
    out.write_text(card, encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    train()
