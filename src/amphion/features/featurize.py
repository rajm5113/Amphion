"""Sequence featurization: amino-acid composition + physicochemical descriptors.

Cheap, interpretable, biology-motivated features (29 total) — the verified
featurizer from notebook 01. Deliberately simple (no deep learning); ESM-2
embeddings are the optional GPU upgrade path.

Features (order == ``FEATURE_NAMES``):
  - 20 amino-acid composition fractions (one per residue in ACDEFGHIKLMNPQRSTVWY)
  - length, net_charge, charge_density, mean_hydrophobicity (Kyte-Doolittle),
    frac_hydrophobic, frac_polar, frac_pos, frac_neg, aromaticity
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..utils import CANONICAL_AA, CANONICAL_SET

# Kyte-Doolittle hydropathy index.
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_PHYS_NAMES = [
    "length", "net_charge", "charge_density", "mean_hydrophobicity",
    "frac_hydrophobic", "frac_polar", "frac_pos", "frac_neg", "aromaticity",
]
FEATURE_NAMES: list[str] = list(CANONICAL_AA) + _PHYS_NAMES
N_FEATURES = len(FEATURE_NAMES)

_HYDROPHOBIC = "AILMFWVC"
_POLAR = "STNQ"
_AROMATIC = "FWY"


def featurize(seq: str) -> np.ndarray:
    """Return the 29-dim feature vector for one peptide (validated, uppercased)."""
    s = str(seq).strip().upper()
    if not s or any(ch not in CANONICAL_SET for ch in s):
        raise ValueError(f"non-canonical residue in sequence: {seq!r}")
    L = len(s)
    c = Counter(s)
    aac = [c.get(a, 0) / L for a in CANONICAL_AA]
    charge = c.get("K", 0) + c.get("R", 0) + 0.1 * c.get("H", 0) - c.get("D", 0) - c.get("E", 0)
    phys = [
        L,
        charge,
        charge / L,
        sum(KD[a] for a in s) / L,
        sum(c.get(a, 0) for a in _HYDROPHOBIC) / L,
        sum(c.get(a, 0) for a in _POLAR) / L,
        (c.get("K", 0) + c.get("R", 0) + c.get("H", 0)) / L,
        (c.get("D", 0) + c.get("E", 0)) / L,
        sum(c.get(a, 0) for a in _AROMATIC) / L,
    ]
    return np.asarray(aac + phys, dtype=float)


def featurize_many(seqs) -> np.ndarray:
    """Stack feature vectors for an iterable of sequences -> (n, 29) array."""
    return np.vstack([featurize(s) for s in seqs])
