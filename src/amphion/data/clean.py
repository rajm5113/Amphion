"""Cleaning + MIC aggregation for the raw sources.

Conventions (from the build spec, Section 5):
  - Alphabet: the 20 standard amino acids only; drop anything else.
  - Length: keep MINLEN..MAXLEN residues.
  - MIC: GRAMPA ``value`` is log10(MIC in uM). Aggregate a peptide's potency as
    the MINIMUM across all tested strains (broad-spectrum best activity).
"""

from __future__ import annotations

import pandas as pd

from ..utils import CANONICAL_SET, get_logger

log = get_logger("amphion.data.clean")


def is_canonical(seq: str) -> bool:
    """True iff every character is one of the 20 standard amino acids."""
    return bool(seq) and all(c in CANONICAL_SET for c in seq)


def normalize_seq(seq) -> str:
    return str(seq).strip().upper()


def clean_sequences(seqs, minlen: int, maxlen: int) -> list[str]:
    """Uppercase, keep canonical + length-bounded, dedupe (order-preserving)."""
    out: list[str] = []
    seen: set[str] = set()
    for s in seqs:
        s = normalize_seq(s)
        if s in seen:
            continue
        if is_canonical(s) and minlen <= len(s) <= maxlen:
            out.append(s)
            seen.add(s)
    return out


def load_grampa(path) -> pd.DataFrame:
    """Load GRAMPA, uppercasing sequences. Expects columns: sequence, value, unit, bacterium, database."""
    g = pd.read_csv(path)
    g["sequence"] = g["sequence"].map(normalize_seq)
    return g


def aggregate_min_mic(grampa: pd.DataFrame, minlen: int, maxlen: int):
    """Return (per_sequence, per_strain).

    per_sequence: one row per unique canonical peptide with
        min_log_mic_uM, min_mic_uM, n_measurements, length.
    per_strain: the cleaned per-measurement rows (sequence, bacterium, log_mic_uM).
    """
    mask = grampa["sequence"].map(lambda s: is_canonical(s) and minlen <= len(s) <= maxlen)
    g = grampa[mask].copy()
    g["log_mic_uM"] = g["value"].astype(float)

    per_seq = (
        g.groupby("sequence")
        .agg(min_log_mic_uM=("log_mic_uM", "min"), n_measurements=("log_mic_uM", "size"))
        .reset_index()
    )
    per_seq["min_mic_uM"] = 10.0 ** per_seq["min_log_mic_uM"]
    per_seq["length"] = per_seq["sequence"].str.len()

    bact_col = "bacterium" if "bacterium" in g.columns else None
    cols = ["sequence", "log_mic_uM"] + ([bact_col] if bact_col else [])
    per_strain = g[cols].rename(columns={bact_col: "bacterium"} if bact_col else {})

    log.info(
        "GRAMPA: %d rows -> %d canonical measurements -> %d unique peptides",
        len(grampa), len(g), len(per_seq),
    )
    return per_seq, per_strain
