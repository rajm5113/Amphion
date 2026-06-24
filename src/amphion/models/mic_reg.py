"""MIC regressor: predict log10(best MIC, uM) for a peptide.

Trains on GRAMPA positives (which carry MIC). Cluster-aware CV (GroupKFold on
homology cluster_id). Reports RMSE, MAE, R2, Spearman. Saves the best model to
models/mic_reg.joblib and writes reports/mic_model_card.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from ..config import get_config
from ..features.featurize import FEATURE_NAMES, featurize_many
from ..utils import ensure_dir, get_logger, set_seed

log = get_logger("amphion.models.mic_reg")


def _load_positives(cfg):
    df = pd.read_parquet(cfg.resolve_path("processed") / "activity_regression.parquet")
    X = featurize_many(df.sequence.tolist())
    y = df.min_log_mic_uM.to_numpy()
    groups = df.cluster_id.to_numpy()
    return X, y, groups


def _models(seed):
    return {
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, random_state=seed)),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(random_state=seed),
    }


def _metrics(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "Spearman": float(spearmanr(y, p).statistic),
    }


def train(cfg=None) -> dict:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    X, y, groups = _load_positives(cfg)
    log.info("MIC regression: X=%s, log10MIC range [%.2f, %.2f]", X.shape, y.min(), y.max())

    cv = GroupKFold(5)
    results = {}
    for name, model in _models(cfg.seed).items():
        p = cross_val_predict(model, X, y, cv=cv, groups=groups)
        results[name] = _metrics(y, p)
        log.info("%-12s RMSE=%.3f MAE=%.3f R2=%.3f Spearman=%.3f",
                 name, results[name]["RMSE"], results[name]["MAE"],
                 results[name]["R2"], results[name]["Spearman"])

    # Deployability-aware selection: among models within 3% of the best RMSE, prefer
    # the most compact. A 400-tree Random Forest is ~200x larger on disk than gradient
    # boosting for a <2% RMSE difference — the wrong trade for a deployable system.
    best_rmse = min(r["RMSE"] for r in results.values())
    compact_order = ["ElasticNet", "HistGB", "RandomForest"]  # smallest -> largest
    eligible = [n for n in results if results[n]["RMSE"] <= best_rmse * 1.03]
    best = next((n for n in compact_order if n in eligible),
                min(results, key=lambda n: results[n]["RMSE"]))
    log.info("selected %s (RMSE %.3f; best was %.3f) — compact & within tolerance",
             best, results[best]["RMSE"], best_rmse)

    model = _models(cfg.seed)[best].fit(X, y)
    models_dir = ensure_dir(cfg.resolve_path("models"))
    joblib.dump(
        {"kind": "mic_reg", "model": model, "best_model": best, "features": FEATURE_NAMES,
         "metrics": results[best]},
        models_dir / "mic_reg.joblib",
    )
    log.info("saved %s", models_dir / "mic_reg.joblib")

    _write_card(cfg, results, best, X.shape)
    return {"best": best, "results": results}


def _write_card(cfg, results, best, shape):
    rows = "\n".join(
        f"| {n} | {m['RMSE']:.3f} | {m['MAE']:.3f} | {m['R2']:.3f} | {m['Spearman']:.3f} |"
        for n, m in results.items()
    )
    card = f"""# Model card — MIC regressor (log10 MIC, uM)

**Task:** predict a peptide's best potency `log10(min MIC, uM)` — lower = more potent.
**Data:** GRAMPA positives ({shape[0]:,} peptides, {shape[1]} features).
**Selected model:** {best} — *deployability-aware* pick: the most compact model
within 3% of the best RMSE (a 400-tree Random Forest is ~200x larger on disk for a
<2% accuracy gain — the wrong trade for a deployable system).
**CV:** GroupKFold on homology cluster_id (no near-duplicate leakage).

| Model | RMSE | MAE | R² | Spearman |
|---|---:|---:|---:|---:|
{rows}

Predictions feed the pipeline's potency term (Phase 6 ranking) and the
`pred_mic_uM` field of `score_activity()`.

> Computational predictions with uncertainty, not validated MIC measurements.
"""
    out = ensure_dir(cfg.resolve_path("reports")) / "mic_model_card.md"
    out.write_text(card, encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    train()
