# Model card — Toxicity classifier (hemolysis)

**Task:** predict whether a peptide is hemolytic (ruptures human red blood cells).
**Data:** HemoPI2 (Raghava lab), 1,926 peptides, 29 features.
**Label:** hemolytic if **HC50 ≤ 100 µM** (lower HC50 = lyses RBCs at lower dose
= more toxic). This threshold reproduces HemoPI2's own binary labels 100%.
**Base rate:** 46.3% hemolytic (mild imbalance → PR-AUC reported).
**Selected model:** RandomForest (best cluster-aware PR-AUC), isotonic-calibrated.

## Cross-validated performance (cluster-aware, StratifiedGroupKFold)
| Model | ROC-AUC | PR-AUC | F1 | MCC |
|---|---:|---:|---:|---:|
| LogReg | 0.746 | 0.661 | 0.681 | 0.400 |
| RandomForest | 0.762 | 0.704 | 0.665 | 0.386 |
| HistGB | 0.763 | 0.694 | 0.662 | 0.386 |

PR-AUC is reported alongside ROC-AUC because the classes are imbalanced; a random
baseline PR-AUC equals the base rate (0.46).

## Sanity probes (calibrated hemolytic probability)
| Peptide | hemolytic_prob |
|---|---:|
| Melittin (hemolytic) | 0.79 |
| Magainin-2 (low hemolysis) | 0.62 |
| Polar control | 0.30 |

Melittin (a classic hemolytic peptide) should score high; benign controls low.

## Use in the pipeline
`score_toxicity(seq) → {hemolytic_prob}`. The assembly loop (Phase 6) drops any
candidate with `hemolytic_prob > toxicity.hemolytic_prob_max`
(default 0.5; lower it to be stricter).

> Computational predictions with uncertainty, not validated safety results.
