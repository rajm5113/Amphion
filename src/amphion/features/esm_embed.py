"""ESM-2 protein-language-model embeddings (the hybrid's activity + toxicity backend).

Loads an ESM-2 model lazily (cached) and returns mean-pooled residue embeddings,
matching exactly how the ESM models were trained in notebooks/02b_esm2_upgrade.ipynb.
Requires torch + fair-esm (the repo's ``[gpu]`` extra; also used by the demo).

The ESM-2 weights download from the PyTorch hub on first use (~600 MB for the
150M model) and are cached under ~/.cache/torch/hub.
"""

from __future__ import annotations

import numpy as np

_CACHE: dict = {}


def _load_esm(model_id: str):
    if model_id not in _CACHE:
        import esm  # fair-esm
        import torch

        model, alphabet = getattr(esm.pretrained, model_id)()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.eval().to(device)
        _CACHE[model_id] = (model, alphabet, alphabet.get_batch_converter(), device)
    return _CACHE[model_id]


def embed_sequences(seqs, esm_cfg: dict, batch_size: int = 16) -> np.ndarray:
    """Mean-pooled ESM-2 embeddings for ``seqs`` -> (n, embed_dim) float32 array.

    ``esm_cfg`` = {model_id, repr_layer, embed_dim, pooling} (stored in each ESM artifact).
    """
    import torch

    model, alphabet, batch_converter, device = _load_esm(esm_cfg["model_id"])
    layer = esm_cfg["repr_layer"]
    seqs = list(seqs)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            data = [(str(j), s) for j, s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            for k, s in enumerate(chunk):
                out.append(rep[k, 1:len(s) + 1].mean(0).float().cpu().numpy())  # exclude BOS/EOS/pad
    return np.asarray(out, dtype=np.float32)
