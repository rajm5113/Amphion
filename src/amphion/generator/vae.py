"""A small character-level sequence VAE over the 20-amino-acid alphabet.

Encoder (GRU) -> latent z -> Decoder (GRU, autoregressive). Trained on AMP
positives (Phase 4 GPU notebook). The interface is deliberately model-agnostic:
a stronger generator (GAN, ESM-2 fine-tune) can drop in behind the same
``sample`` / checkpoint contract.

Requires PyTorch (a GPU phase). The repo installs torch only via the optional
``[gpu]`` extra; the assembled CPU pipeline consumes the generated candidates.csv
rather than importing this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: 3 special tokens + the 20 standard amino acids.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, SOS, EOS = 0, 1, 2
ITOS = ["<pad>", "<sos>", "<eos>"] + list(CANONICAL_AA)
STOI = {c: i for i, c in enumerate(ITOS)}
VOCAB = len(ITOS)  # 23


def encode_seq(seq: str, max_residues: int) -> list[int]:
    """Sequence -> token ids: [SOS, ...aa, EOS] padded with PAD to max_residues+2."""
    toks = [SOS] + [STOI[c] for c in seq] + [EOS]
    toks = toks[: max_residues + 2]
    toks += [PAD] * (max_residues + 2 - len(toks))
    return toks


def decode_tokens(tokens) -> str:
    """Token ids -> amino-acid string (stops at EOS; ignores special tokens)."""
    out = []
    for t in tokens:
        t = int(t)
        if t == EOS:
            break
        if t >= 3:  # an amino acid
            out.append(ITOS[t])
    return "".join(out)


class PeptideVAE(nn.Module):
    def __init__(self, vocab=VOCAB, embed=64, hidden=256, latent=32, max_residues=50):
        super().__init__()
        self.hparams = {
            "vocab": vocab, "embed": embed, "hidden": hidden,
            "latent": latent, "max_residues": max_residues,
        }
        self.max_residues = max_residues
        self.emb = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.enc_gru = nn.GRU(embed, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.lat2hid = nn.Linear(latent, hidden)
        self.dec_gru = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def encode(self, x):
        _, h = self.enc_gru(self.emb(x))           # h: (1, B, hidden)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, x_in):
        h0 = torch.tanh(self.lat2hid(z)).unsqueeze(0)
        out, _ = self.dec_gru(self.emb(x_in), h0)
        return self.out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x[:, :-1])          # teacher forcing
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, n, device, temperature=1.0, max_residues=None):
        """Sample ``n`` sequences from the prior. Returns a list of AA strings."""
        self.eval()
        max_residues = max_residues or self.max_residues
        z = torch.randn(n, self.hparams["latent"], device=device)
        h = torch.tanh(self.lat2hid(z)).unsqueeze(0)
        tok = torch.full((n, 1), SOS, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        seqs = [[] for _ in range(n)]
        for _ in range(max_residues + 1):
            out, h = self.dec_gru(self.emb(tok), h)
            logits = self.out(out[:, -1]) / max(temperature, 1e-6)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            for i in range(n):
                if not done[i]:
                    if nxt[i].item() == EOS:
                        done[i] = True
                    elif nxt[i].item() >= 3:
                        seqs[i].append(ITOS[int(nxt[i].item())])
            tok = nxt.unsqueeze(1)
            if done.all():
                break
        return ["".join(s) for s in seqs]


def vae_loss(logits, target, mu, logvar, beta=1.0):
    """Reconstruction CE (ignoring PAD) + beta * KL divergence. Returns (loss, ce, kl)."""
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ignore_index=PAD, reduction="mean",
    )
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return ce + beta * kl, ce, kl
