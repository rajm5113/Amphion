# Model card — Activity classifier (AMP vs non-AMP)

**Task:** predict whether a peptide is antimicrobial, from sequence alone.
**Data:** balanced, length-matched subset (6,698 peptides, 29 features).
**Selected model:** HistGB (chosen by cluster-aware AUC), isotonic-calibrated.

## Cross-validated performance
Two CV schemes. **Cluster-aware** (group by homology cluster) is the honest number;
random split is shown only to reproduce the known-good baseline.

| Model | Random AUC | Cluster-aware AUC | Acc | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| LogReg | 0.910 | 0.902 | 0.827 | 0.828 | 0.654 |
| RandomForest | 0.971 | 0.952 | 0.888 | 0.885 | 0.777 |
| HistGB | 0.971 | 0.954 | 0.890 | 0.888 | 0.781 |

> Random-split RF AUC ≈ 0.97 reproduces the baseline; the cluster-aware AUC is
> lower and more honest — it forbids near-duplicate sequences leaking across folds.

## Biological sanity check ✅ PASS
Top features should be charge-related (AMPs are cationic and disrupt the
negatively-charged bacterial membrane). Charge-related features in the top 6:
**net_charge, charge_density, K, frac_pos**.

| Feature | Importance |
|---|---:|
| net_charge | 0.143 |
| charge_density | 0.110 |
| K | 0.065 |
| frac_pos | 0.064 |
| M | 0.058 |
| frac_neg | 0.048 |
| frac_polar | 0.042 |
| frac_hydrophobic | 0.040 |
| R | 0.035 |
| mean_hydrophobicity | 0.033 |
| length | 0.032 |
| D | 0.028 |

## Known-peptide check (calibrated probability)
| Peptide | amp_prob |
|---|---:|
| Magainin-2 (AMP) | 0.98 |
| Melittin (AMP) | 0.99 |
| Random non-AMP | 0.03 |

Known AMPs should score ≈ 1.0, the random control low.

## Calibration & use
Probabilities are isotonic-calibrated so `amp_prob` is a usable confidence.
Full reliability diagrams + Brier score come in Phase 7.

> These are computational predictions with uncertainty, not validated results.
