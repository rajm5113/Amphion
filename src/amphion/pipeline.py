"""End-to-end assembly: candidates -> validity -> novelty -> activity -> toxicity
-> gates -> rank -> shortlist. All thresholds/weights are config-driven.

Consumes ``data/interim/candidates.csv`` (the generator's output, produced on the
Kaggle GPU). Writes ``reports/shortlist.csv`` + ``reports/shortlist_report.md`` with
full per-candidate provenance (every sub-score) and an uncertainty estimate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import get_config
from .features.featurize import featurize_many
from .filters.novelty import load_all_known, score_novelty
from .rank import add_rank_scores
from .utils import CANONICAL_SET, ensure_dir, get_logger

log = get_logger("amphion.pipeline")


def _batch_score(seqs, cfg) -> pd.DataFrame:
    """Score all sequences in one pass (hybrid backend): amp_prob (+uncertainty),
    pred MIC, hemolytic_prob. ESM embeddings are computed once and reused."""
    from .score import _load, _load_preferred

    clf = _load_preferred("activity_clf_esm.joblib", "activity_clf.joblib")
    reg = _load("mic_reg.joblib")  # hybrid: MIC stays physicochemical
    tox = _load_preferred("tox_clf_esm.joblib", "tox_clf.joblib")

    arts = (clf, reg, tox)
    X_phys = featurize_many(seqs) if any(a.get("features") != "esm" for a in arts) else None
    X_esm = None
    if any(a.get("features") == "esm" for a in arts):
        from .features.esm_embed import embed_sequences

        esm_cfg = next(a["esm"] for a in arts if a.get("features") == "esm")
        X_esm = embed_sequences(seqs, esm_cfg, batch_size=32)

    def feat(a):
        return X_esm if a.get("features") == "esm" else X_phys

    cal = clf["model"]
    Xa = feat(clf)
    amp_prob = cal.predict_proba(Xa)[:, 1]
    try:  # ensemble spread across the calibrated CV sub-models = uncertainty
        sub = np.stack([cc.predict_proba(Xa)[:, 1] for cc in cal.calibrated_classifiers_], axis=1)
        amp_unc = sub.std(axis=1)
    except Exception:
        amp_unc = np.full(len(seqs), np.nan)

    pred_log_mic = reg["model"].predict(feat(reg))
    hemolytic = tox["model"].predict_proba(feat(tox))[:, 1]
    return pd.DataFrame({
        "sequence": list(seqs),
        "length": [len(s) for s in seqs],
        "amp_prob": amp_prob,
        "amp_uncertainty": amp_unc,
        "pred_log_mic": pred_log_mic,
        "pred_mic_uM": 10.0 ** pred_log_mic,
        "hemolytic_prob": hemolytic,
    })


def run(n=None, candidates_csv=None, out_csv=None, cfg=None) -> pd.DataFrame:
    cfg = cfg or get_config()
    interim = cfg.resolve_path("interim")
    reports = ensure_dir(cfg.resolve_path("reports"))
    candidates_csv = Path(candidates_csv or interim / "candidates.csv")
    out_csv = Path(out_csv or reports / "shortlist.csv")

    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"{candidates_csv} not found — run the Phase-4 Kaggle notebook and drop "
            f"candidates.csv into data/interim/ first."
        )

    cand = pd.read_csv(candidates_csv)
    if n:
        cand = cand.head(n)

    # 1) validity gate (canonical, length, unique)
    lo, hi = cfg.length.min, cfg.length.max
    seqs = [s.upper() for s in cand["sequence"].astype(str)]
    seqs = [s for s in seqs if s and all(c in CANONICAL_SET for c in s) and lo <= len(s) <= hi]
    seqs = list(dict.fromkeys(seqs))
    log.info("candidates: %d in file -> %d valid & unique", len(cand), len(seqs))

    # 2) novelty (vs ALL known) and 3) activity + toxicity scoring
    nov = score_novelty(seqs, known=load_all_known(cfg), cfg=cfg)
    scored = _batch_score(seqs, cfg)
    df = scored.merge(
        nov[["sequence", "max_identity", "nearest_known", "novelty_score", "novel"]],
        on="sequence", how="left",
    )

    # 4) flow gates: active enough, safe enough, novel
    g_act = df["amp_prob"] >= cfg.activity.amp_prob_min
    g_tox = df["hemolytic_prob"] <= cfg.toxicity.hemolytic_prob_max
    g_nov = df["novel"].fillna(False)
    df["passes_gates"] = g_act & g_tox & g_nov
    log.info("gates -> active:%d  safe:%d  novel:%d  ALL:%d",
             int(g_act.sum()), int(g_tox.sum()), int(g_nov.sum()), int(df["passes_gates"].sum()))

    # 5) rank survivors
    survivors = df[df["passes_gates"]].copy()
    ranked = add_rank_scores(survivors, cfg) if len(survivors) else survivors

    cols = [
        "sequence", "length", "rank_score", "amp_prob", "amp_uncertainty",
        "pred_mic_uM", "pred_log_mic", "hemolytic_prob", "novelty_score",
        "max_identity", "nearest_known", "potency_term", "safety_term", "synthesizability",
    ]
    ranked = ranked.reindex(columns=[c for c in cols if c in ranked.columns or c in survivors.columns])
    ranked.to_csv(out_csv, index=False)
    log.info("wrote shortlist: %d ranked candidates -> %s", len(ranked), out_csv)

    _write_report(cfg, ranked, df, reports / "shortlist_report.md")
    return ranked


def _write_report(cfg, ranked, allscored, out: Path):
    n_valid = len(allscored)
    n_short = len(ranked)
    top = ranked.head(20)

    def fmt_rows(t):
        out = []
        for i, r in t.iterrows():
            out.append(
                f"| {i+1} | `{r['sequence']}` | {r['rank_score']:.3f} | {r['amp_prob']:.2f}"
                f"±{r.get('amp_uncertainty', float('nan')):.2f} | {r['pred_mic_uM']:.1f} | "
                f"{r['hemolytic_prob']:.2f} | {r['novelty_score']:.2f} | {r['synthesizability']:.2f} |"
            )
        return "\n".join(out)

    body = (
        "_No candidate passed all three gates with the current thresholds. "
        "Loosen them in `config.yaml` (e.g. lower `activity.amp_prob_min` or raise "
        "`novelty.max_identity_to_known`) and re-run._"
        if n_short == 0 else
        f"""| # | sequence | rank | amp_prob | pred MIC (µM) | hemolytic | novelty | synth |
|---:|---|---:|---:|---:|---:|---:|---:|
{fmt_rows(top)}"""
    )

    md = f"""# Amphion shortlist — novel, predicted-active, predicted-safe candidates

**Pipeline:** generate → validity → novelty → activity → toxicity → rank (config-driven).

| Stage | Count |
|---|---:|
| Valid unique candidates scored | {n_valid:,} |
| Passed activity gate (amp_prob ≥ {cfg.activity.amp_prob_min}) | {int((allscored['amp_prob'] >= cfg.activity.amp_prob_min).sum()):,} |
| Passed safety gate (hemolytic_prob ≤ {cfg.toxicity.hemolytic_prob_max}) | {int((allscored['hemolytic_prob'] <= cfg.toxicity.hemolytic_prob_max).sum()):,} |
| Passed novelty gate (identity < {cfg.novelty.max_identity_to_known}) | {int(allscored['novel'].fillna(False).sum()):,} |
| **Shortlist (passed ALL gates)** | **{n_short:,}** |

## Top candidates
Ranked by the composite score (weights from `config.yaml`:
amp {cfg.ranking.w_amp}, potency {cfg.ranking.w_potency}, safety {cfg.ranking.w_safety},
novelty {cfg.ranking.w_novelty}, synth {cfg.ranking.w_synth}). `amp_prob` shown with its
ensemble uncertainty (±). Full provenance for every candidate is in `shortlist.csv`.

{body}

## How to read this
- **rank_score** — weighted blend of the columns to its right; higher = a better all-round candidate.
- **amp_prob ± unc** — calibrated probability the peptide is antimicrobial, with ensemble uncertainty.
- **pred MIC (µM)** — predicted best potency; lower = more potent (regressor is the weakest model — treat as a coarse guide).
- **hemolytic** — predicted probability it harms human red blood cells; lower = safer.
- **novelty** — 1 − (max identity to any known sequence); higher = more original.
- **synth** — heuristic ease-of-synthesis (length / charge / cysteine / hydrophobicity).

> **These are computational predictions with uncertainty, not validated hits.** No
> software confirms a real bacterial kill or human-cell safety — that requires a wet
> lab (Stage 3, out of scope). This shortlist is a prioritized starting point for
> experimental validation, nothing more.
"""
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s", out)
