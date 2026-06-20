"""Homology de-duplication via greedy single-linkage clustering.

Why: near-identical sequences split across train/test inflate scores. We assign
each sequence a ``cluster_id`` so Phase 2 can do *cluster-aware* cross-validation
(no near-duplicate leaks between folds).

Approach: a CD-HIT-like greedy pass. Process sequences longest-first; assign each
to the first existing representative it is >= ``identity_threshold`` similar to,
else start a new cluster with it as representative. Deterministic given the input.

Identity proxy: rapidfuzz ``fuzz.ratio`` (normalized indel similarity, 0-100).
For exact biological identity, MMseqs2/CD-HIT are better; this is the free,
dependency-light fallback the spec calls for and is sufficient to catch
near-duplicates. Falls back to exact-match-only clustering if rapidfuzz is absent.
"""

from __future__ import annotations

from ..utils import get_logger

log = get_logger("amphion.data.cluster")


def greedy_cluster(sequences, identity_threshold: float = 0.6) -> dict[str, int]:
    """Return a mapping ``sequence -> cluster_id`` (ints from 0)."""
    uniq = sorted(set(sequences), key=lambda s: (-len(s), s))  # longest-first, deterministic

    try:
        from rapidfuzz import fuzz, process
    except Exception:  # pragma: no cover - rapidfuzz is a hard dep, but degrade gracefully
        log.warning("rapidfuzz unavailable -> exact-match clustering only")
        return {s: i for i, s in enumerate(uniq)}

    cutoff = identity_threshold * 100.0
    reps: list[str] = []
    rep_cluster: list[int] = []
    assign: dict[str, int] = {}
    next_id = 0

    for s in uniq:
        match = process.extractOne(s, reps, scorer=fuzz.ratio, score_cutoff=cutoff) if reps else None
        if match is not None:
            assign[s] = rep_cluster[match[2]]
        else:
            assign[s] = next_id
            reps.append(s)
            rep_cluster.append(next_id)
            next_id += 1

    log.info(
        "%d unique sequences -> %d clusters (identity >= %.2f)",
        len(uniq), next_id, identity_threshold,
    )
    return assign
