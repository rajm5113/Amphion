"""Novelty filter: how different is each candidate from everything already known?

For each candidate we compute the maximum sequence identity against the UNION of
all known sequences (AMP positives + non-AMP negatives + hemolysis set) — not just
the positives, per the spec. A candidate is **novel** if its closest match is below
``novelty.max_identity_to_known`` (default 0.6 = 60% identity).

Identity proxy: rapidfuzz ``fuzz.ratio`` (normalized indel similarity), fast in C.
"""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz, process

from ..config import get_config
from ..utils import get_logger

log = get_logger("amphion.filters.novelty")


def load_all_known(cfg=None) -> list[str]:
    """Every sequence Amphion has seen, across all training sources."""
    cfg = cfg or get_config()
    processed = cfg.resolve_path("processed")
    known: set[str] = set()
    clf = processed / "activity_classification.parquet"
    hemo = processed / "hemolysis.parquet"
    if clf.exists():
        known |= set(pd.read_parquet(clf, columns=["sequence"]).sequence)
    if hemo.exists():
        known |= set(pd.read_parquet(hemo, columns=["sequence"]).sequence)
    return sorted(known)


def score_novelty(candidates, known=None, threshold=None, cfg=None) -> pd.DataFrame:
    """Annotate each candidate with max_identity, nearest_known, novelty_score, novel.

    novelty_score = 1 - max_identity (higher = more novel).
    novel = max_identity < threshold.
    """
    cfg = cfg or get_config()
    threshold = cfg.novelty.max_identity_to_known if threshold is None else threshold
    known = list(known) if known is not None else load_all_known(cfg)
    cand = list(candidates)
    log.info("scoring novelty: %d candidates vs %d known sequences (threshold=%.2f)",
             len(cand), len(known), threshold)

    rows = []
    for i, c in enumerate(cand):
        m = process.extractOne(c, known, scorer=fuzz.ratio)  # (seq, score 0-100, idx)
        ident = (m[1] / 100.0) if m else 0.0
        rows.append((c, ident, m[0] if m else None))
        if (i + 1) % 1000 == 0:
            log.info("  %d / %d", i + 1, len(cand))

    df = pd.DataFrame(rows, columns=["sequence", "max_identity", "nearest_known"])
    df["novelty_score"] = 1.0 - df["max_identity"]
    df["novel"] = df["max_identity"] < threshold
    log.info("novel candidates: %d / %d (%.0f%%)", int(df.novel.sum()), len(df), 100 * df.novel.mean())
    return df


if __name__ == "__main__":
    cfg = get_config()
    cands = pd.read_csv(cfg.resolve_path("interim") / "candidates.csv").sequence.tolist()
    out = score_novelty(cands, cfg=cfg)
    print(out.head(10).to_string(index=False))
