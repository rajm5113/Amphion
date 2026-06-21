"""Phase 7 — evaluation, benchmark reproduction, calibration, honesty.

Produces reports/evaluation.md + figures:
  (a) headline metrics vs published AMP/hemolysis classifiers (cited),
  (b) recovery test: do known potent AMPs score above random decoys,
  (c) calibration: reliability diagram + Brier score (cluster-aware holdout),
  (d) applicability-domain check for the generated candidates.

Run:  .venv/Scripts/python scripts/benchmark.py
"""

from __future__ import annotations

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

from amphion import get_config, get_logger, score_activity, set_seed
from amphion.features.featurize import featurize_many
from amphion.models.activity_clf import _models

log = get_logger("amphion.benchmark")

# Well-characterized natural/clinical AMPs — the model should recover these as active.
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


def _random_peptides(n, rng, lo=5, hi=50):
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    out = []
    for _ in range(n):
        L = int(rng.integers(lo, hi + 1))
        out.append("".join(rng.choice(aa, L)))
    return out


def benchmark_table(cfg) -> str:
    models = cfg.resolve_path("models")
    act = joblib.load(models / "activity_clf.joblib")
    tox = joblib.load(models / "tox_clf.joblib")
    reg = joblib.load(models / "mic_reg.joblib")
    am = act["metrics"]["cluster_aware"]
    tm = tox["metrics"]
    rm = reg["metrics"]
    return f"""| Model | Our metric (cluster-aware CV) | Published reference (own splits) |
|---|---|---|
| Activity (AMP vs non-AMP) | ROC-AUC **{am['AUC']:.3f}**, MCC {am['MCC']:.3f} | AMP Scanner v2 ≈0.96, AMPlify/iAMPpred ≈0.88–0.93 [1,2] |
| Hemolysis | ROC-AUC **{tm['ROC_AUC']:.3f}**, PR-AUC {tm['PR_AUC']:.3f} | HemoPI ≈0.95 (random negs), HemoPI2 ≈0.86 [3,4] |
| MIC regression | RMSE **{rm['RMSE']:.2f}** log10 µM, Spearman {rm['Spearman']:.2f} | sequence-only MIC models report R²≈0.2–0.5 [5] |

Our numbers use **cluster-aware** cross-validation (no near-duplicate leakage between
folds), which is stricter than the random splits most published numbers use — so a
slightly lower number here can still be the more honest one."""


def recovery_test(cfg, figdir) -> dict:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    known_probs = {n: score_activity(s)["amp_prob"] for n, s in KNOWN_AMPS.items()}
    decoys = _random_peptides(500, rng)
    decoy_probs = featurize_then_amp(cfg, decoys)

    kp = np.array(list(known_probs.values()))
    recovery = float((kp >= cfg.activity.amp_prob_min).mean())
    y = np.r_[np.ones(len(kp)), np.zeros(len(decoy_probs))]
    p = np.r_[kp, decoy_probs]
    auc = roc_auc_score(y, p)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(decoy_probs, bins=25, alpha=0.6, label=f"random decoys (n={len(decoy_probs)})", density=True)
    ax.hist(kp, bins=10, alpha=0.7, label=f"known AMPs (n={len(kp)})", density=True)
    ax.axvline(cfg.activity.amp_prob_min, color="k", ls="--", lw=1)
    ax.set_xlabel("amp_prob"); ax.set_ylabel("density")
    ax.set_title("Recovery: known AMPs vs random decoys"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_recovery.png", dpi=110); plt.close(fig)
    return {"known_probs": known_probs, "recovery": recovery, "auc": auc,
            "decoy_mean": float(np.mean(decoy_probs))}


def featurize_then_amp(cfg, seqs) -> np.ndarray:
    act = joblib.load(cfg.resolve_path("models") / "activity_clf.joblib")
    return act["model"].predict_proba(featurize_many(seqs))[:, 1]


def calibration_test(cfg, figdir) -> dict:
    set_seed(cfg.seed)
    df = pd.read_parquet(cfg.resolve_path("processed") / "activity_classification.parquet")
    df = df[df.in_balanced].reset_index(drop=True)
    X = featurize_many(df.sequence.tolist()); y = df.label.to_numpy(); groups = df.cluster_id.to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.seed).split(X, y, groups))
    best = joblib.load(cfg.resolve_path("models") / "activity_clf.joblib")["base_model"]
    cal = CalibratedClassifierCV(_models(cfg.seed)[best], method="isotonic", cv=5).fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])[:, 1]
    brier = brier_score_loss(y[te], p)
    frac_pos, mean_pred = calibration_curve(y[te], p, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=f"{best} (Brier={brier:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (held-out, cluster-aware)"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_calibration.png", dpi=110); plt.close(fig)
    return {"brier": float(brier), "model": best}


def applicability_domain(cfg, figdir) -> dict:
    clf = pd.read_parquet(cfg.resolve_path("processed") / "activity_classification.parquet")
    Xtr = featurize_many(clf.sequence.tolist())
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    nn = NearestNeighbors(n_neighbors=2).fit(Xtr_s)
    d_tr = nn.kneighbors(Xtr_s)[0][:, 1]
    thr = float(np.percentile(d_tr, 99))

    short = pd.read_csv(cfg.resolve_path("reports") / "shortlist.csv")
    Xc_s = scaler.transform(featurize_many(short.sequence.tolist()))
    d_c = nn.kneighbors(Xc_s, n_neighbors=1)[0][:, 0]
    inside = float((d_c <= thr).mean())

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(d_tr, bins=50, alpha=0.6, density=True, label="training")
    ax.hist(d_c, bins=50, alpha=0.6, density=True, label="shortlist candidates")
    ax.axvline(thr, color="crimson", ls="--", label="99th pct of training")
    ax.set_xlabel("distance to nearest training peptide (standardized feature space)")
    ax.set_ylabel("density"); ax.set_title("Applicability domain"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "eval_applicability.png", dpi=110); plt.close(fig)
    return {"inside_fraction": inside, "threshold": thr}


def main():
    cfg = get_config()
    figdir = cfg.resolve_path("figures")
    log.info("benchmark table..."); table = benchmark_table(cfg)
    log.info("recovery test..."); rec = recovery_test(cfg, figdir)
    log.info("calibration..."); cal = calibration_test(cfg, figdir)
    log.info("applicability domain..."); ad = applicability_domain(cfg, figdir)

    known_rows = "\n".join(f"| {n} | {p:.2f} |" for n, p in rec["known_probs"].items())
    md = f"""# Amphion — evaluation, benchmarks & honest limitations

## 1. Headline metrics vs published methods
{table}

## 2. Recovery test — does it know real AMPs from noise?
Scored {len(KNOWN_AMPS)} well-characterized AMPs against 500 random peptides.
- **Recovery:** {rec['recovery']*100:.0f}% of known AMPs scored ≥ {cfg.activity.amp_prob_min} (active).
- **Separation:** ROC-AUC **{rec['auc']:.3f}** distinguishing known AMPs from random decoys
  (decoy mean amp_prob {rec['decoy_mean']:.2f}).

> Honest caveat: several of these classic AMPs (e.g. Magainin-2, Melittin, LL-37) are in
> GRAMPA training data, so this is a **sanity check that the model behaves correctly**, not
> a generalization claim. The cluster-aware CV in §1 is the generalization number.

![recovery](figures/eval_recovery.png)

| Known AMP | amp_prob |
|---|---:|
{known_rows}

## 3. Calibration — is `amp_prob` a trustworthy confidence?
Held-out, cluster-aware: isotonic-calibrated **{cal['model']}**, **Brier score {cal['brier']:.3f}**
(lower is better; 0.25 = uninformative). Points near the diagonal = well-calibrated.

![calibration](figures/eval_calibration.png)

## 4. Applicability domain — are the candidates in-distribution?
Each candidate's distance to its nearest training peptide (standardized features).
**{ad['inside_fraction']*100:.0f}%** of the shortlist lies within the training distribution
(≤ the 99th percentile of training-to-training distances). Candidates beyond it are
extrapolations and should be treated with extra caution.

![applicability](figures/eval_applicability.png)

## 5. Uncertainty
Every shortlisted candidate carries an `amp_uncertainty` (spread across the calibrated
CV ensemble) in `shortlist.csv` — a per-candidate confidence band on `amp_prob`.

## 6. Limitations (read this)
- **Predictions ≠ proof.** No software confirms a real bacterial kill or human-cell
  safety. Physical MIC/hemolysis validation is Stage 3 (wet lab), out of scope.
- **Negatives are a modeling choice.** Activity negatives are UniProt decoys (answers
  "is this an AMP at all?"), not low-potency peptides. Different negatives → different model.
- **Distribution shift.** The generator can propose peptides unlike anything in training;
  the applicability-domain flag (§4) is a guard, not a guarantee.
- **MIC regression is weak** (R² modest); treat `pred_mic_uM` as a coarse ranking aid,
  not a quantitative prediction.
- **Hemolysis model is the weakest link** (ROC-AUC ≈0.76); simple composition features
  under-capture toxicity. ESM-2 embeddings are the documented upgrade path.
- **Generator posterior collapse.** The VAE's KL term collapsed (decoder ignored the
  latent), so some raw samples echoed training motifs — the novelty filter removes these,
  but a stronger generator (β-tuning / free-bits / fine-tuned protein LM) would improve diversity.

> Every number here is a computational prediction with uncertainty. The value of Amphion
> is a *prioritized, transparent shortlist* for experimental testing — not validated drugs.
"""
    out = cfg.resolve_path("reports") / "evaluation.md"
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s + 3 figures", out)
    print(f"recovery {rec['recovery']*100:.0f}% | recovery AUC {rec['auc']:.3f} | "
          f"Brier {cal['brier']:.3f} | AD inside {ad['inside_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
