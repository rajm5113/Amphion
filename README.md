# AMP Designer — AI-Designed Antimicrobial Peptides

An end-to-end, **computer-only** pipeline that designs novel antimicrobial peptide (AMP) candidates, predicts which are likely to kill bacteria, filters out those likely to harm human cells, and ranks the survivors into a shortlist.

> **Why this matters:** antibiotic resistance causes over a million deaths a year. AMPs — short peptides that kill bacteria by disrupting their membranes — are a promising class because resistance to them is harder to evolve. This project builds the AI design-and-screening half of the discovery process.
>
> **Honest scope:** outputs are *predictions with uncertainty*, not proof. Confirming a real kill needs a wet lab (out of scope here). This is a research-grade portfolio project and the artifact you'd show a lab to recruit a Stage-3 collaborator.

## Pipeline (target)
```
public AMP databases -> clean/label -> (A) activity predictor   } score
                                        (B) toxicity predictor   } + filter
generator -> novel candidates -> novelty filter -> A -> B -> rank -> shortlist
```

## Status
| Layer | What | Status |
|------|------|--------|
| **1** | Data + baseline activity classifier (AMP vs non-AMP) | ✅ **Done** |
| 2 | MIC regression + toxicity/hemolysis filter | ▢ Next |
| 3 | Generator (VAE / small protein LM) | ▢ |
| 4 | Assemble loop: generate → novelty → activity → toxicity → rank | ▢ |

## Layer 1 results
- **Data:** GRAMPA (consolidated DBAASP + DRAMP + YADAMP + APD + DADP; ~6.8k peptides, ~51k MIC measurements) for positives; AMPlify UniProt-derived non-AMPs for negatives. Balanced and length-matched.
- **Features:** amino-acid composition + physicochemical descriptors (charge, hydrophobicity, etc.).
- **Model:** Random Forest — **5-fold CV ROC-AUC 0.973**, accuracy 0.92, MCC 0.85 (held-out test AUC 0.974). Competitive with published AMP classifiers.
- **Sanity check:** top predictive features are *net charge* and *cationic residues* — the model independently recovered the real biophysical mechanism of AMPs. Scores Magainin-2 and Melittin ≈ 1.0, random sequences ≈ 0.1.

## Run it
Open `01_data_and_baseline_classifier.ipynb` in Kaggle or Colab (internet ON), Run All. CPU is fine; trains in under a minute. Produces `amp_activity_baseline.joblib`.

## Data sources
GRAMPA (github.com/zswitten/Antimicrobial-Peptides) · AMPlify (github.com/BirolLab/AMPlify) · DBAASP (dbaasp.org) · DRAMP · APD3

## License / use
Research and educational use. All data sources are open-access; cite them in any writeup.
