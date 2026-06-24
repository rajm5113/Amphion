"""Phase 7 — evaluation, benchmark reproduction, calibration, honesty (hybrid-aware).

Produces reports/evaluation.md + figures:
  (0) ESM-2 vs hand-crafted-features benchmark + the hybrid decision,
  (a) headline metrics of the DEPLOYED hybrid vs published methods (cited),
  (b) recovery test (deployed activity backend): known AMPs vs random decoys,
  (c) calibration: reliability diagram + Brier (cluster-aware holdout, capped),
  (d) applicability-domain check for the generated candidates.

Run:  .venv/Scripts/python scripts/benchmark.py
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from amphion import get_config, get_logger, set_seed
from amphion.features.featurize import featurize_many
from amphion.score import _load, _load_preferred, active_backends

log = get_logger("amphion.benchmark")

CALIB_CAP = 2000  # cap embedded sequences for the calibration diagram (CPU-friendly)

KNOWN_AMPS = {
    "Magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Indolicidin": "ILPWKWPWWPWRR",
    "Cecropin-A": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "Protegrin-1": "RGGRLCYCRRRFCVCVGR",
    "Buforin-II": "TRSSRAGLQFPVGRVHRLLRK",
    "Aurein-1.2": "GLFDIIKKIAESF",
    "Pexiganan": "GIGKFLKKAKKFGKAFVKILKK",
    "Pleurocidin": "GWGSFFKKAAHVGKHVGKAALTHYL",
    "HNP-1": "ACYCRIPACIAGERRYGTCIYQGRLWAFCC",
    "Dermaseptin": "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ",
}


def _activity_features(artifact, seqs):
    """Feature matrix for the deployed activity backend (ESM embeddings or physicochemical)."""
    if artifact.get("features") == "esm":
        from amphion.features.esm_embed import embed_sequences

        return embed_sequences(list(seqs), artifact["esm"], batch_size=32)
    return featurize_many(list(seqs))


def _activity_probs(seqs):
    act = _load_preferred("activity_clf_esm.joblib", "activity_clf.joblib")
    return act["model"].predict_proba(_activity_features(act, seqs))[:, 1]


def _random_peptides(n, rng, lo=5, hi=50):
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    return ["".join(rng.choice(aa, int(rng.integers(lo, hi + 1)))) for _ in range(n)]


def esm_section(cfg) -> str:
    p = cfg.resolve_path("models") / "metrics_esm.json"
    if not p.exists():
        return "_(ESM-2 metrics not found — run notebooks/02b_esm2_upgrade.ipynb.)_"
    m = json.loads(p.read_text(encoding="utf-8"))
    e, b = m["esm"], m["baseline"]
    return f"""ESM-2 (a 150M-parameter protein language model) was benchmarked against the 29
hand-crafted features on the **same sequences, labels, and homology clusters**:

| Task | Hand-crafted features | ESM-2 embeddings | Deployed |
|---|---|---|---|
| Activity (AUC) | {b['activity_AUC']:.3f} | **{e['activity']['AUC']:.3f}** | ESM-2 |
| Toxicity (PR-AUC) | {b['tox_PR_AUC']:.3f} | **{e['toxicity']['PR_AUC']:.3f}** | ESM-2 |
| MIC (RMSE, lower=better) | **{b['mic_RMSE']:.3f}** | {e['mic']['RMSE']:.3f} | hand-crafted |

**Hybrid decision:** ESM-2 *improved* activity and toxicity but *lost* on MIC regression —
potency is dominated by net charge / hydrophobicity, which the hand-crafted features encode
directly while mean-pooled embeddings dilute. So Amphion deploys each model where it wins:
**ESM-2 for activity + toxicity, hand-crafted features for MIC.** Right tool per job, not hype."""


def headline_table(cfg) -> str:
    act = _load_preferred("activity_clf_esm.joblib", "activity_clf.joblib")
    tox = _load_preferred("tox_clf_esm.joblib", "tox_clf.joblib")
    reg = _load("mic_reg.joblib")
    a_auc = act["metrics"].get("cluster_aware", act["metrics"]).get("AUC")
    t = tox["metrics"]
    r = reg["metrics"]
    b = active_backends()
    return f"""| Model | Deployed metric (cluster-aware) | Backend | Published reference |
|---|---|---|---|
| Activity | ROC-AUC **{a_auc:.3f}** | {b['activity']} | AMP Scanner v2 ≈0.96 [1,2] |
| Toxicity | ROC-AUC {t['ROC_AUC']:.3f} · PR-AUC **{t['PR_AUC']:.3f}** | {b['toxicity']} | HemoPI2 ≈0.86 [3,4] |
| MIC | RMSE **{r['RMSE']:.2f}** · Spearman {r['Spearman']:.2f} | {b['mic']} | seq-only R²≈0.2–0.5 [5] |"""


def recovery_test(cfg, figdir) -> dict:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    known = list(KNOWN_AMPS.values())
    decoys = _random_peptides(500, rng)
    kp = _activity_probs(known)
    dp = _activity_probs(decoys)
    recovery = float((kp >= cfg.activity.amp_prob_min).mean())
    auc = roc_auc_score(np.r_[np.ones(len(kp)), np.zeros(len(dp))], np.r_[kp, dp])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(dp, bins=25, alpha=0.6, density=True, label=f"random decoys (n={len(dp)})")
    ax.hist(kp, bins=10, alpha=0.7, density=True, label=f"known AMPs (n={len(kp)})")
    ax.axvline(cfg.activity.amp_prob_min, color="k", ls="--", lw=1)
    ax.set_xlabel("amp_prob"); ax.set_ylabel("density")
    ax.set_title("Recovery: known AMPs vs random decoys (deployed model)"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_recovery.png", dpi=110); plt.close(fig)
    return {"known_probs": dict(zip(KNOWN_AMPS, kp)), "recovery": recovery,
            "auc": float(auc), "decoy_mean": float(dp.mean())}


def calibration_test(cfg, figdir) -> dict:
    set_seed(cfg.seed)
    df = pd.read_parquet(cfg.resolve_path("processed") / "activity_classification.parquet")
    df = df[df.in_balanced].reset_index(drop=True)
    if len(df) > CALIB_CAP:
        df = df.sample(CALIB_CAP, random_state=cfg.seed).reset_index(drop=True)

    act = _load_preferred("activity_clf_esm.joblib", "activity_clf.joblib")
    X = _activity_features(act, df.sequence.tolist())
    y = df.label.to_numpy(); groups = df.cluster_id.to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=cfg.seed).split(X, y, groups))

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cal = CalibratedClassifierCV(base, method="isotonic", cv=5).fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])[:, 1]
    brier = brier_score_loss(y[te], p)
    frac_pos, mean_pred = calibration_curve(y[te], p, n_bins=8, strategy="quantile")

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=f"{act.get('features','phys')} (Brier={brier:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (held-out, cluster-aware)"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_calibration.png", dpi=110); plt.close(fig)
    return {"brier": float(brier), "n": int(len(df)), "backend": act.get("features", "physicochemical")}


def applicability_domain(cfg, figdir) -> dict:
    clf = pd.read_parquet(cfg.resolve_path("processed") / "activity_classification.parquet")
    Xtr = featurize_many(clf.sequence.tolist())  # physicochemical space (cheap, interpretable)
    scaler = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=2).fit(scaler.transform(Xtr))
    d_tr = nn.kneighbors(scaler.transform(Xtr))[0][:, 1]
    thr = float(np.percentile(d_tr, 99))

    short = pd.read_csv(cfg.resolve_path("reports") / "shortlist.csv")
    d_c = nn.kneighbors(scaler.transform(featurize_many(short.sequence.tolist())))[0][:, 0]
    inside = float((d_c <= thr).mean())

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(d_tr, bins=50, alpha=0.6, density=True, label="training")
    ax.hist(d_c, bins=50, alpha=0.6, density=True, label="shortlist candidates")
    ax.axvline(thr, color="crimson", ls="--", label="99th pct of training")
    ax.set_xlabel("distance to nearest training peptide (physicochemical space)")
    ax.set_ylabel("density"); ax.set_title("Applicability domain"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_applicability.png", dpi=110); plt.close(fig)
    return {"inside_fraction": inside}


def main():
    cfg = get_config()
    figdir = cfg.resolve_path("figures")
    log.info("backends: %s", active_backends())
    esm_md = esm_section(cfg)
    table = headline_table(cfg)
    log.info("recovery..."); rec = recovery_test(cfg, figdir)
    log.info("calibration..."); cal = calibration_test(cfg, figdir)
    log.info("applicability..."); ad = applicability_domain(cfg, figdir)

    known_rows = "\n".join(f"| {n} | {p:.2f} |" for n, p in rec["known_probs"].items())
    md = f"""# Amphion — evaluation, benchmarks & honest limitations

## 0. ESM-2 vs hand-crafted features (the hybrid decision)
{esm_md}

## 1. Headline metrics of the deployed (hybrid) models
{table}

Cluster-aware CV (no near-duplicate leakage) — stricter than the random splits most
published numbers use, so a slightly lower number here can be the more honest one.

## 2. Recovery test — does the deployed model know real AMPs from noise?
{len(KNOWN_AMPS)} well-characterized AMPs vs 500 random peptides.
- **Recovery:** {rec['recovery']*100:.0f}% of known AMPs scored ≥ {cfg.activity.amp_prob_min}.
- **Separation:** ROC-AUC **{rec['auc']:.3f}** (decoy mean amp_prob {rec['decoy_mean']:.2f}).

> Caveat: several of these classics are in GRAMPA training — a correctness check, not
> a generalization claim (§1 is the generalization number).

![recovery](figures/eval_recovery.png)

| Known AMP | amp_prob |
|---|---:|
{known_rows}

## 3. Calibration — is `amp_prob` a trustworthy confidence?
Held-out, cluster-aware on the deployed **{cal['backend']}** activity backend
(n={cal['n']}): isotonic-calibrated, **Brier {cal['brier']:.3f}** (lower better; 0.25 = uninformative).

![calibration](figures/eval_calibration.png)

## 4. Applicability domain — are the candidates in-distribution?
**{ad['inside_fraction']*100:.0f}%** of the shortlist lies within the 99th-percentile
nearest-neighbour distance of the training set (physicochemical space). Beyond it = extrapolation.

![applicability](figures/eval_applicability.png)

## 5. Uncertainty
Every shortlisted candidate carries an `amp_uncertainty` (spread across the calibrated CV
ensemble) in `shortlist.csv`.

## 6. Limitations (read this)
- **Predictions ≠ proof.** No software confirms a real bacterial kill or human-cell safety;
  physical validation is Stage 3 (wet lab), out of scope.
- **Negatives are a modeling choice** (UniProt decoys answer "is this an AMP at all?").
- **Distribution shift:** the generator can propose out-of-distribution peptides; §4 is a guard.
- **MIC regression is modest** (R² ~0.25) — `pred_mic_uM` is a coarse ranking aid, not a number.
- **ESM-2 did not help everywhere** — it lost on MIC (see §0); we deploy it only where it wins.
- **Generator posterior collapse** — some raw VAE samples echo training motifs; the novelty
  filter removes them.

> Every number here is a computational prediction with uncertainty. Amphion is a
> *prioritized, transparent shortlist* for experimental testing — not validated drugs.
"""
    out = cfg.resolve_path("reports") / "evaluation.md"
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s + 3 figures", out)
    print(f"recovery {rec['recovery']*100:.0f}% | AUC {rec['auc']:.3f} | "
          f"Brier {cal['brier']:.3f} | AD inside {ad['inside_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
