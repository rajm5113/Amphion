"""Build the hemolysis dataset -> data/processed/hemolysis.parquet.

Source: HemoPI2 (Raghava lab) — columns SEQUENCE, HC50 (uM), label.

Label definition (spec-required, documented): a peptide is **hemolytic** if its
HC50 <= ``toxicity.hemolytic_hc50_uM_max`` uM (default 100). Lower HC50 = lyses
red blood cells at lower concentration = more toxic. This threshold reproduces
HemoPI2's own binary labels 100% on its data (verified), so it is well calibrated.

Duplicate sequences are collapsed to their **minimum** HC50 (conservative: if any
measurement says toxic, treat as toxic).

Output columns:
  sequence, hc50_uM, log10_hc50_uM, hemolytic (0/1), length, cluster_id, source_split
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import get_config
from ..utils import ensure_dir, get_logger, hash_sequences, sha256_file, set_seed, write_json
from .clean import is_canonical, normalize_seq
from .cluster import greedy_cluster
from .download import download_toxicity

log = get_logger("amphion.data.build_tox")


def _read_hemopi2(path, split: str, minlen: int, maxlen: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["sequence", "hc50_uM", "label_orig"][: len(df.columns)]
    df["sequence"] = df["sequence"].map(normalize_seq)
    df["hc50_uM"] = pd.to_numeric(df["hc50_uM"], errors="coerce")
    df["source_split"] = split
    mask = df["sequence"].map(lambda s: is_canonical(s) and minlen <= len(s) <= maxlen)
    return df[mask & df["hc50_uM"].notna()].copy()


def build(cfg=None, force_download: bool = False) -> dict:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    minlen, maxlen = cfg.length.min, cfg.length.max
    hc50_thr = cfg.toxicity.hemolytic_hc50_uM_max
    processed = ensure_dir(cfg.resolve_path("processed"))

    paths = download_toxicity(cfg, force=force_download)
    raw = pd.concat(
        [
            _read_hemopi2(paths["hemopi2_crossval"], "crossval", minlen, maxlen),
            _read_hemopi2(paths["hemopi2_independent"], "independent", minlen, maxlen),
        ],
        ignore_index=True,
    )

    # collapse duplicate sequences -> minimum HC50 (conservative / safety-first)
    idx = raw.groupby("sequence")["hc50_uM"].idxmin()
    df = raw.loc[idx].reset_index(drop=True)

    # label from HC50 threshold; verify agreement with the source's own label
    df["hemolytic"] = (df["hc50_uM"] <= hc50_thr).astype(int)
    df["log10_hc50_uM"] = np.log10(df["hc50_uM"])
    df["length"] = df["sequence"].str.len()
    agree = float((df["hemolytic"] == df["label_orig"].astype(int)).mean()) if "label_orig" in df else float("nan")

    clusters = greedy_cluster(df["sequence"].tolist(), identity_threshold=cfg.clustering.identity_threshold)
    df["cluster_id"] = df["sequence"].map(clusters).astype(int)

    df = df[
        ["sequence", "hc50_uM", "log10_hc50_uM", "hemolytic", "length", "cluster_id", "source_split"]
    ].sort_values("sequence").reset_index(drop=True)

    out = processed / "hemolysis.parquet"
    df.to_parquet(out, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "HemoPI2 (raghavagps) cross_val + independent",
        "params": {
            "seed": cfg.seed,
            "length": {"min": minlen, "max": maxlen},
            "hemolytic_hc50_uM_max": hc50_thr,
            "hemolytic_log10_hc50_threshold": round(math.log10(hc50_thr), 4),
            "duplicate_policy": "min HC50 (conservative)",
        },
        "counts": {
            "peptides": int(len(df)),
            "hemolytic": int(df.hemolytic.sum()),
            "non_hemolytic": int((df.hemolytic == 0).sum()),
            "clusters": int(df.cluster_id.max() + 1),
            "label_agreement_with_source": round(agree, 4),
        },
        "content_hash": hash_sequences(df.sequence),
        "file_sha256": sha256_file(out),
    }
    write_json(processed / "manifest_tox.json", manifest)
    log.info(
        "hemolysis: %d peptides (%d hemolytic / %d non), %d clusters, label agreement %.3f -> %s",
        len(df), int(df.hemolytic.sum()), int((df.hemolytic == 0).sum()),
        int(df.cluster_id.max() + 1), agree, out.name,
    )
    return manifest


if __name__ == "__main__":
    import json

    print(json.dumps(build(), indent=2))
