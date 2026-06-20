"""Build the clean, leakage-free, labeled datasets persisted to ``data/processed/``.

Outputs
-------
activity_classification.parquet
    One row per unique peptide. Columns:
      sequence, label (1=AMP, 0=non-AMP), source, length, cluster_id, in_balanced
    The FULL imbalanced set is kept; ``in_balanced`` flags the length-matched 1:1
    subset used to train the baseline classifier.
activity_regression.parquet
    Positives only (they carry MIC). Columns:
      sequence, min_log_mic_uM, min_mic_uM, n_measurements, active, length, cluster_id
mic_per_strain.parquet
    Cleaned per-measurement rows: sequence, bacterium, log_mic_uM.
manifest.json
    Row counts, content hashes, parameters, file hashes — for reproducibility.

Labels
------
Active (config ``activity.active_mic_uM_max``, default 32 uM): a peptide is
"active" if its best MIC <= 32 uM, i.e. min_log_mic_uM <= log10(32) = 1.505.
Alternative threshold <=25 ug/mL would need per-peptide MW conversion (not done).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import ensure_dir, get_logger, hash_sequences, read_fasta, sha256_file, set_seed, write_json
from .clean import aggregate_min_mic, clean_sequences, load_grampa
from .cluster import greedy_cluster
from .download import download_all

log = get_logger("amphion.data.build")


def _length_matched_balanced(pos: list[str], neg: list[str], seed: int, bin_size: int = 5):
    """Pick a 1:1, length-matched (in ``bin_size``-residue bins) subset. Returns (sel_pos, sel_neg)."""
    rng = np.random.default_rng(seed)
    pb, nb = defaultdict(list), defaultdict(list)
    for s in pos:
        pb[len(s) // bin_size].append(s)
    for s in neg:
        nb[len(s) // bin_size].append(s)
    sel_pos, sel_neg = [], []
    for b in sorted(set(pb) | set(nb)):
        k = min(len(pb[b]), len(nb[b]))
        if k:
            sel_pos += list(rng.choice(sorted(pb[b]), k, replace=False))
            sel_neg += list(rng.choice(sorted(nb[b]), k, replace=False))
    return sel_pos, sel_neg


def build(cfg=None, force_download: bool = False) -> dict:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    minlen, maxlen = cfg.length.min, cfg.length.max
    processed = ensure_dir(cfg.resolve_path("processed"))

    # 1) acquire ---------------------------------------------------------
    paths = download_all(cfg, force=force_download)

    # 2) positives: GRAMPA unique peptides with min MIC ------------------
    grampa = load_grampa(paths["grampa"])
    per_seq, per_strain = aggregate_min_mic(grampa, minlen, maxlen)
    pos = per_seq["sequence"].tolist()
    pos_set = set(pos)

    # 3) negatives: AMPlify non-AMPs, cleaned, minus any leakage ---------
    neg_raw = read_fasta(paths["neg_train"]) + read_fasta(paths["neg_test"])
    neg = [s for s in clean_sequences(neg_raw, minlen, maxlen) if s not in pos_set]
    log.info("positives=%d  negatives=%d (leakage-free)", len(pos), len(neg))

    # 4) homology clusters over the UNION (global, cluster-aware splits) -
    clusters = greedy_cluster(pos + neg, identity_threshold=cfg.clustering.identity_threshold)

    # 5) balanced, length-matched 1:1 subset -----------------------------
    sel_pos, sel_neg = _length_matched_balanced(pos, neg, cfg.seed)
    balanced = set(sel_pos) | set(sel_neg)
    log.info("balanced length-matched subset: %d AMPs + %d non-AMPs", len(sel_pos), len(sel_neg))

    # 6) classification dataset (full, with balanced flag) ---------------
    rows = []
    for s in pos:
        rows.append(("__P__", s, 1, "grampa"))
    for s in neg:
        rows.append(("__N__", s, 0, "amplify"))
    clf = pd.DataFrame(rows, columns=["_k", "sequence", "label", "source"]).drop(columns="_k")
    clf["length"] = clf["sequence"].str.len()
    clf["cluster_id"] = clf["sequence"].map(clusters).astype(int)
    clf["in_balanced"] = clf["sequence"].isin(balanced)
    clf = clf.sort_values(["label", "sequence"]).reset_index(drop=True)

    # 7) regression dataset (positives only) -----------------------------
    log_thr = math.log10(cfg.activity.active_mic_uM_max)
    reg = per_seq.copy()
    reg["active"] = reg["min_log_mic_uM"] <= log_thr
    reg["cluster_id"] = reg["sequence"].map(clusters).astype(int)
    reg = reg[
        ["sequence", "min_log_mic_uM", "min_mic_uM", "n_measurements", "active", "length", "cluster_id"]
    ].sort_values("sequence").reset_index(drop=True)

    # 8) persist ---------------------------------------------------------
    f_clf = processed / "activity_classification.parquet"
    f_reg = processed / "activity_regression.parquet"
    f_strain = processed / "mic_per_strain.parquet"
    clf.to_parquet(f_clf, index=False)
    reg.to_parquet(f_reg, index=False)
    per_strain.reset_index(drop=True).to_parquet(f_strain, index=False)

    # 9) manifest --------------------------------------------------------
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "seed": cfg.seed,
            "length": {"min": minlen, "max": maxlen},
            "active_mic_uM_max": cfg.activity.active_mic_uM_max,
            "active_log10_mic_threshold": round(log_thr, 4),
            "cluster_identity_threshold": cfg.clustering.identity_threshold,
        },
        "counts": {
            "grampa_rows": int(len(grampa)),
            "positives_unique": len(pos),
            "negatives_unique": len(neg),
            "clusters_total": int(max(clusters.values()) + 1) if clusters else 0,
            "classification_rows": int(len(clf)),
            "balanced_rows": int(clf["in_balanced"].sum()),
            "regression_rows": int(len(reg)),
            "active_positives": int(reg["active"].sum()),
            "inactive_positives": int((~reg["active"]).sum()),
            "per_strain_rows": int(len(per_strain)),
        },
        "content_hashes": {
            "positives": hash_sequences(pos),
            "negatives": hash_sequences(neg),
            "balanced": hash_sequences(balanced),
        },
        "files": {
            f_clf.name: sha256_file(f_clf),
            f_reg.name: sha256_file(f_reg),
            f_strain.name: sha256_file(f_strain),
        },
    }
    write_json(processed / "manifest.json", manifest)
    log.info("wrote %s, %s, %s + manifest.json", f_clf.name, f_reg.name, f_strain.name)
    return manifest


if __name__ == "__main__":
    import json

    print(json.dumps(build(), indent=2))
