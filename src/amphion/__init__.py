"""Amphion — an AI pipeline that designs novel antimicrobial peptide (AMP)
candidates, predicts activity, filters toxicity, and ranks survivors.

Honest scope: every output is a *prediction with uncertainty*, never proof.
A real bacterial kill must be confirmed in a wet lab (Stage 3, out of scope).
"""

from .config import load_config, get_config, repo_root, Config
from .utils import set_seed, get_logger, ensure_dir, read_fasta
from .score import score_activity

__version__ = "0.1.0"

__all__ = [
    "load_config",
    "get_config",
    "repo_root",
    "Config",
    "set_seed",
    "get_logger",
    "ensure_dir",
    "read_fasta",
    "score_activity",
    "__version__",
]
