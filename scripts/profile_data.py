"""Profile the processed datasets -> reports/data_profile.md (+ figures).

Run:  .venv/Scripts/python scripts/profile_data.py
Reads data/processed/*.parquet (run amphion.data.build_datasets first).
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")  # headless: save figures, no display
import matplotlib.pyplot as plt
import pandas as pd

from amphion import get_config, get_logger
from amphion.utils import ensure_dir

log = get_logger("amphion.profile")


def main():
    cfg = get_config()
    processed = cfg.resolve_path("processed")
    figdir = ensure_dir(cfg.resolve_path("figures"))
    reports = ensure_dir(cfg.resolve_path("reports"))

    clf = pd.read_parquet(processed / "activity_classification.parquet")
    reg = pd.read_parquet(processed / "activity_regression.parquet")
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))

    pos = clf[clf.label == 1]
    neg = clf[clf.label == 0]
    log_thr = math.log10(cfg.activity.active_mic_uM_max)

    # --- figure 1: length distribution, positives vs negatives ----------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = range(cfg.length.min, cfg.length.max + 2)
    ax.hist(pos.length, bins=bins, alpha=0.6, label=f"AMP (n={len(pos)})")
    ax.hist(neg.length, bins=bins, alpha=0.6, label=f"non-AMP (n={len(neg)})")
    ax.set_xlabel("peptide length (residues)")
    ax.set_ylabel("count")
    ax.set_title("Length distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "data_length_dist.png", dpi=110)
    plt.close(fig)

    # --- figure 2: MIC distribution with active threshold ---------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(reg.min_log_mic_uM, bins=60)
    ax.axvline(log_thr, color="crimson", ls="--", label=f"active <= {cfg.activity.active_mic_uM_max} uM")
    ax.set_xlabel("log10(best MIC, uM) — lower = more potent")
    ax.set_ylabel("count (unique peptides)")
    ax.set_title("Potency (best MIC) distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "data_mic_dist.png", dpi=110)
    plt.close(fig)

    # --- figure 3: cluster size distribution ----------------------------
    sizes = clf.cluster_id.value_counts().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set_xlabel("cluster size (sequences per homology cluster)")
    ax.set_ylabel("number of clusters")
    ax.set_title("Homology cluster sizes")
    fig.tight_layout()
    fig.savefig(figdir / "data_cluster_sizes.png", dpi=110)
    plt.close(fig)

    # --- leakage check --------------------------------------------------
    overlap = set(pos.sequence) & set(neg.sequence)

    c = manifest["counts"]
    p = manifest["params"]
    md = f"""# Data profile — Amphion Phase 1

_Generated from `data/processed/` by `scripts/profile_data.py`. All figures in `reports/figures/`._

## Overview

| Quantity | Value |
|---|---:|
| GRAMPA raw MIC measurements | {c['grampa_rows']:,} |
| Unique canonical peptides (positives) | {c['positives_unique']:,} |
| AMPlify non-AMP negatives (leakage-free) | {c['negatives_unique']:,} |
| Classification rows (full, imbalanced) | {c['classification_rows']:,} |
| Balanced + length-matched subset | {c['balanced_rows']:,} |
| Homology clusters | {c['clusters_total']:,} |
| Active positives (best MIC ≤ {p['active_mic_uM_max']} µM) | {c['active_positives']:,} |
| Inactive positives | {c['inactive_positives']:,} |

**Leakage check:** sequences appearing in both classes = **{len(overlap)}** (must be 0). ✅

## Labels & conventions
- **Positives:** unique GRAMPA peptides, potency = `min(log10 MIC µM)` across all tested strains (broad-spectrum best activity).
- **Negatives:** AMPlify UniProt-derived non-AMPs, cleaned to the 20 standard amino acids, length {p['length']['min']}–{p['length']['max']}.
- **Active label:** best MIC ≤ {p['active_mic_uM_max']} µM, i.e. `min_log_mic_uM ≤ {p['active_log10_mic_threshold']}`. (Alternative ≤25 µg/mL would need per-peptide MW conversion — not applied.)
- **Homology clustering:** greedy single-linkage at identity ≥ {p['cluster_identity_threshold']}; `cluster_id` stored per sequence for **cluster-aware CV** in Phase 2.

## Class balance & length
The balanced subset is matched 1:1 in 5-residue length bins so the classifier cannot cheat on length alone.

![length distribution](figures/data_length_dist.png)

## Potency (MIC) distribution
{c['active_positives']:,} of {c['positives_unique']:,} positives ({100*c['active_positives']/c['positives_unique']:.0f}%) are "active" at the ≤ {p['active_mic_uM_max']} µM threshold.

![MIC distribution](figures/data_mic_dist.png)

## Homology clusters
{c['clusters_total']:,} clusters across {c['classification_rows']:,} sequences. Cluster-aware splits prevent near-duplicate leakage between train and test.

![cluster sizes](figures/data_cluster_sizes.png)

## Files
| File | Rows | SHA-256 (first 12) |
|---|---:|---|
| `activity_classification.parquet` | {c['classification_rows']:,} | `{manifest['files']['activity_classification.parquet'][:12]}` |
| `activity_regression.parquet` | {c['regression_rows']:,} | `{manifest['files']['activity_regression.parquet'][:12]}` |
| `mic_per_strain.parquet` | {c['per_strain_rows']:,} | `{manifest['files']['mic_per_strain.parquet'][:12]}` |

_Built {manifest['created_utc']} · seed {p['seed']}._

> Reminder: these are computational predictions with uncertainty, not validated results.
"""
    out = reports / "data_profile.md"
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s and 3 figures", out)
    print(f"data_profile.md written; leakage overlap = {len(overlap)} (expect 0)")


if __name__ == "__main__":
    main()
