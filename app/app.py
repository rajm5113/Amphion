"""Amphion demo — paste a peptide, get activity / toxicity / novelty + a gate verdict.

    .venv/Scripts/python app/app.py        # launches a local Gradio server

Hugging Face Spaces-ready (see app/README.md). Requires the trained model artifacts
in models/ (train them via the README Quickstart, or commit them to the Space).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `amphion` package importable whether app.py sits in app/ (repo) or at
# the Space root. Look for a sibling or parent `src/amphion`.
_here = Path(__file__).resolve().parent
for _cand in (_here / "src", _here.parent / "src"):
    if (_cand / "amphion").is_dir():
        sys.path.insert(0, str(_cand))
        break

import gradio as gr

from amphion import get_config, score_activity, score_toxicity
from amphion.filters.novelty import score_novelty
from amphion.utils import CANONICAL_SET

cfg = get_config()

EXAMPLES = [
    ["TAVVKVKVKVSDKLAKGPVSIGPSIVVNHAHEALKKL"],  # top Amphion-designed candidate — PASSES all gates
    ["GIGKFLHSAKKFGKAFVGEIMNS"],      # Magainin-2 — active but a known peptide (not novel)
    ["GIGAVLKVLTTGLPALISWIKRKRQQ"],   # Melittin — active but hemolytic
    ["SEEGDTAATGGDSTGAESDTAAGSE"],    # polar control — inactive
]


def evaluate(seq: str) -> str:
    seq = (seq or "").strip().upper()
    if not seq:
        return "Enter a peptide sequence (5–50 standard amino acids)."
    bad = sorted(set(seq) - CANONICAL_SET)
    if bad:
        return f"❌ **Invalid**: non-standard residue(s) {bad}. Use only ACDEFGHIKLMNPQRSTVWY."
    if not (cfg.length.min <= len(seq) <= cfg.length.max):
        return f"❌ **Length {len(seq)} out of range** [{cfg.length.min}, {cfg.length.max}]."

    try:
        a = score_activity(seq)
        t = score_toxicity(seq)
    except FileNotFoundError:
        return "⚠️ Trained models not found. Train them first (see the README Quickstart)."

    nov = score_novelty([seq], cfg=cfg).iloc[0]
    g_act = a["amp_prob"] >= cfg.activity.amp_prob_min
    g_tox = t["hemolytic_prob"] <= cfg.toxicity.hemolytic_prob_max
    g_nov = bool(nov["novel"])
    ok = g_act and g_tox and g_nov

    def mark(b):
        return "✅" if b else "❌"

    verdict = ("## ✅ Passes all gates — a shortlist-worthy candidate"
               if ok else "## ❌ Filtered out (fails ≥1 gate below)")
    return f"""{verdict}

**`{seq}`**  · length {len(seq)}

| Metric | Value | Gate |
|---|---|---|
| Antimicrobial probability | **{a['amp_prob']:.2f}** | {mark(g_act)} keep ≥ {cfg.activity.amp_prob_min} |
| Predicted MIC (potency) | {a['pred_mic_uM']:.1f} µM | lower = more potent |
| Hemolytic probability | **{t['hemolytic_prob']:.2f}** | {mark(g_tox)} keep ≤ {cfg.toxicity.hemolytic_prob_max} |
| Novelty score | {nov['novelty_score']:.2f} | {mark(g_nov)} novel if identity < {cfg.novelty.max_identity_to_known} |
| Nearest known peptide | `{nov['nearest_known']}` | identity {nov['max_identity']:.2f} |

> These are **computational predictions with uncertainty, not proof**. No software confirms a
> real bacterial kill or human-cell safety — that requires wet-lab validation (out of scope).
"""


demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Textbox(label="Peptide sequence", placeholder="e.g. GIGKFLHSAKKFGKAFVGEIMNS", lines=2),
    outputs=gr.Markdown(label="Amphion assessment"),
    examples=EXAMPLES,
    title="🧬 Amphion — AI-designed antimicrobial peptide screener",
    description=(
        "Paste a peptide (5–50 standard amino acids). Amphion predicts whether it's "
        "**antimicrobial**, its **potency**, whether it's **hemolytic** (toxic to human cells), "
        "and how **novel** it is — then applies the pipeline's gates. Predictions only, not proof."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
