"""Composite ranking of candidates that pass the gates (Section 8 of the spec).

    rank_score = w_amp     * amp_prob
               + w_potency * potency_term        # normalized(-pred_log_mic)
               + w_safety  * (1 - hemolytic_prob)
               + w_novelty * novelty_score
               + w_synth   * synthesizability

All weights live in ``config.yaml`` so the story can be re-tuned without code changes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import get_config
from .features.featurize import KD


def synthesizability(seq: str) -> float:
    """Heuristic ease-of-synthesis score in [0, 1] (1 = easy). Penalizes the things
    that make a peptide hard to make or handle: extreme length, many cysteines
    (disulfide complexity), very high net charge, strong hydrophobicity (aggregation)."""
    L = len(seq)
    c = Counter(seq)
    pen = 0.0
    if L > 30:
        pen += min(0.30, (L - 30) / 20 * 0.30)      # long peptides are harder to synthesize
    if L < 8:
        pen += (8 - L) / 8 * 0.10                    # very short: handling/activity caveats
    nC = c.get("C", 0)
    if nC > 2:
        pen += min(0.30, (nC - 2) * 0.10)            # disulfide complexity
    charge = c.get("K", 0) + c.get("R", 0) + 0.1 * c.get("H", 0) - c.get("D", 0) - c.get("E", 0)
    if abs(charge) > 9:
        pen += min(0.20, (abs(charge) - 9) / 10 * 0.20)
    kd = sum(KD[a] for a in seq) / L
    if kd > 1.5:
        pen += min(0.20, (kd - 1.5) / 2 * 0.20)      # aggregation-prone
    return float(max(0.0, 1.0 - pen))


def add_rank_scores(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Add potency_term, safety_term, synthesizability, rank_score columns.

    Requires columns: sequence, amp_prob, pred_log_mic, hemolytic_prob, novelty_score.
    potency_term is min-max normalized across the provided pool.
    """
    cfg = cfg or get_config()
    w = cfg.ranking
    df = df.copy()

    neg_logmic = -df["pred_log_mic"].to_numpy()
    rng = neg_logmic.max() - neg_logmic.min()
    df["potency_term"] = (neg_logmic - neg_logmic.min()) / (rng + 1e-9)
    df["safety_term"] = 1.0 - df["hemolytic_prob"]
    df["synthesizability"] = df["sequence"].map(synthesizability)

    df["rank_score"] = (
        w.w_amp * df["amp_prob"]
        + w.w_potency * df["potency_term"]
        + w.w_safety * df["safety_term"]
        + w.w_novelty * df["novelty_score"]
        + w.w_synth * df["synthesizability"]
    )
    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)
