# Amphion — AI-Designed Antimicrobial Peptides

**An end-to-end, computer-only pipeline that designs novel antimicrobial peptide (AMP) candidates, predicts which are likely to kill bacteria, filters out those likely to harm human cells, and ranks the survivors into a shortlist.**

> **Why this matters.** Antibiotic resistance kills over a million people a year and the drug pipeline is nearly empty. AMPs — short peptides that kill bacteria by physically disrupting their membranes — are a promising class because resistance to them is harder to evolve. Amphion builds the AI *design-and-screening* half of the discovery process.
>
> **Honest scope.** Every output is a **prediction with uncertainty, not proof**. No software confirms a real bacterial kill or human-cell safety — that needs a wet lab (Stage 3, out of scope here). This is a research-grade portfolio project: the artifact you'd show a lab to recruit a wet-lab collaborator.

## Pipeline
```
public AMP databases ──> clean + label ──┬──> (A) activity model  (kills bacteria?)
                                          └──> (B) toxicity model  (safe for human cells?)

generator ──> novel candidates ──> novelty filter ──> (A) keep active
                                                  ──> (B) drop toxic ──> rank ──> SHORTLIST
```

## Headline results
| Stage | Model | Performance (cluster-aware CV) |
|---|---|---|
| **Activity** (AMP vs non-AMP) | HistGradientBoosting, calibrated | **ROC-AUC 0.95** · random-split 0.97 · biology sanity check ✅ |
| **MIC** (potency, log₁₀ µM) | Random Forest | RMSE 0.76 · Spearman 0.52 |
| **Toxicity** (hemolysis) | Random Forest, class-weighted | ROC-AUC 0.76 · PR-AUC 0.70 |
| **Generator** | char-level sequence VAE (Kaggle GPU) | 5,000 valid candidates; composition matches real AMPs |
| **Shortlist** | full pipeline | **467 novel, predicted-active, predicted-safe candidates** |

**Evaluation:** 100% recovery of 12 classic AMPs · recovery AUC 0.997 vs random decoys · calibration Brier 0.070 · 100% of shortlist within the training distribution. See [`reports/evaluation.md`](reports/evaluation.md).

**Biology sanity check (the result that matters):** the top features driving the activity model are **net charge, charge density, lysine, and fraction-positive** — the model independently recovered the real biophysical mechanism of AMPs (they are cationic and disrupt negatively-charged bacterial membranes). Magainin-2 and Melittin score ≈1.0; random sequences ≈0.1.

| Data profile | Feature importances | Recovery test | Calibration |
|---|---|---|---|
| ![len](reports/figures/data_length_dist.png) | ![mic](reports/figures/data_mic_dist.png) | ![rec](reports/figures/eval_recovery.png) | ![cal](reports/figures/eval_calibration.png) |

## Quickstart
```bash
# 1. Project-local virtual env (all deps stay here — nothing global)
python -m venv .venv
.venv/Scripts/python -m pip install -e .          # Windows
# source .venv/bin/activate && pip install -e .    # macOS/Linux

# 2. Build datasets (downloads public data, ~1 min, CPU)
.venv/Scripts/python -m amphion.data.build_datasets
.venv/Scripts/python -m amphion.data.build_tox

# 3. Train the CPU models (~2 min total)
.venv/Scripts/python -m amphion.models.activity_clf
.venv/Scripts/python -m amphion.models.mic_reg
.venv/Scripts/python -m amphion.models.tox_clf

# 4. (GPU, once) train the generator on Kaggle/Colab:
#    run notebooks/04_train_generator.ipynb, drop generator_vae.pt -> models/
#    and candidates.csv -> data/interim/   (see notebook header for steps)

# 5. Assemble the loop -> ranked shortlist
.venv/Scripts/python -m amphion.run                # writes reports/shortlist.csv

# 6. Evaluation report + figures
.venv/Scripts/python scripts/benchmark.py
```

Score a single peptide in Python:
```python
from amphion import score_activity, score_toxicity
score_activity("GIGKFLHSAKKFGKAFVGEIMNS")   # {'amp_prob': 0.98, 'pred_mic_uM': 1.9, ...}
score_toxicity("GIGAVLKVLTTGLPALISWIKRKRQQ") # {'hemolytic_prob': 0.79}  (Melittin — toxic)
```

Or launch the interactive demo (paste a sequence, see all scores):
```bash
.venv/Scripts/python app/app.py
```

## What's where
| Path | Contents |
|---|---|
| `src/amphion/` | the package: `data/`, `features/`, `models/`, `generator/`, `filters/`, `rank.py`, `pipeline.py` |
| `config.yaml` | every threshold, weight, path, and data URL (the story is tuned here) |
| `notebooks/` | `01` baseline (CPU) · `04_train_generator.ipynb` (Kaggle GPU) |
| `reports/` | `data_profile.md`, model cards, `shortlist.csv` + report, `evaluation.md`, `figures/` |
| `docs/METHODS.md` | concise scientific writeup |
| `app/` | Gradio demo (Hugging Face Spaces-ready) |

## Compute split
Everything runs on a **local CPU** except the generator (Phase 4), which trains on a **free Kaggle/Colab GPU** — a self-contained notebook is provided. All data is open-access and fetched at runtime; no paid APIs, datasets, or compute.

## Data sources
GRAMPA ([github.com/zswitten/Antimicrobial-Peptides](https://github.com/zswitten/Antimicrobial-Peptides)) · AMPlify ([github.com/BirolLab/AMPlify](https://github.com/BirolLab/AMPlify)) · HemoPI2 ([github.com/raghavagps/HemoPI2](https://github.com/raghavagps/HemoPI2)) · DBAASP · DRAMP · AMP Scanner v2 · ESM-2 (optional embeddings).

## Limitations
Predictions are not proof; the MIC and hemolysis models are the weakest links (simple composition features); the VAE shows partial posterior collapse (mitigated by the novelty filter); activity negatives are a modeling choice. Full discussion in [`reports/evaluation.md`](reports/evaluation.md).

## License
Research and educational use. All data sources are open-access; cite them in any writeup.

> Reminder for every report Amphion produces: these are computational predictions with uncertainty, not validated results. A real kill must be confirmed in a wet lab.
