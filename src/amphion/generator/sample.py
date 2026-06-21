"""Generate candidate peptides from a trained VAE and enforce the validity gate.

Validity gate (every candidate must pass before leaving this phase):
  - canonical 20-AA only (guaranteed by the vocabulary),
  - length within [length.min, length.max],
  - not an exact duplicate of any training sequence,
  - unique within the generated batch.

Writes ``data/interim/candidates.csv`` (sequence, length, source). Requires torch;
on Kaggle the Phase-4 notebook calls the equivalent logic inline and returns this CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get_config
from ..utils import ensure_dir, get_logger

log = get_logger("amphion.generator.sample")


def load_vae(model_path, device="cpu"):
    """Rebuild a PeptideVAE from a checkpoint dict {hparams, state_dict}."""
    import torch

    from .vae import PeptideVAE

    ckpt = torch.load(model_path, map_location=device)
    model = PeptideVAE(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _known_training_sequences(cfg) -> set[str]:
    """All sequences the generator must not exactly reproduce (the AMP positives it trained on)."""
    p = cfg.resolve_path("processed") / "activity_regression.parquet"
    if p.exists():
        return set(pd.read_parquet(p).sequence)
    return set()


def generate_candidates(
    n=5000, model_path=None, out=None, temperature=1.0, seed=42,
    oversample=3.0, cfg=None,
):
    """Sample until ``n`` valid, unique, novel candidates are collected; write candidates.csv."""
    import torch

    cfg = cfg or get_config()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(model_path or cfg.resolve_path("models") / "generator_vae.pt")
    out = Path(out or cfg.resolve_path("interim") / "candidates.csv")

    model = load_vae(model_path, device)
    known = _known_training_sequences(cfg)
    lo, hi = cfg.length.min, cfg.length.max

    seen, kept = set(), []
    target_draws = int(n * oversample)
    while len(kept) < n:
        for s in model.sample(target_draws, device, temperature=temperature):
            if lo <= len(s) <= hi and s not in known and s not in seen:
                seen.add(s)
                kept.append(s)
                if len(kept) >= n:
                    break
        log.info("collected %d / %d valid novel candidates", len(kept), n)

    df = pd.DataFrame({"sequence": kept, "length": [len(s) for s in kept], "source": "vae"})
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    log.info("wrote %d candidates -> %s", len(df), out)
    return df


if __name__ == "__main__":
    generate_candidates()
