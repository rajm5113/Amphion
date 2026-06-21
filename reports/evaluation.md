# Amphion — evaluation, benchmarks & honest limitations

## 1. Headline metrics vs published methods
| Model | Our metric (cluster-aware CV) | Published reference (own splits) |
|---|---|---|
| Activity (AMP vs non-AMP) | ROC-AUC **0.954**, MCC 0.781 | AMP Scanner v2 ≈0.96, AMPlify/iAMPpred ≈0.88–0.93 [1,2] |
| Hemolysis | ROC-AUC **0.762**, PR-AUC 0.704 | HemoPI ≈0.95 (random negs), HemoPI2 ≈0.86 [3,4] |
| MIC regression | RMSE **0.76** log10 µM, Spearman 0.52 | sequence-only MIC models report R²≈0.2–0.5 [5] |

Our numbers use **cluster-aware** cross-validation (no near-duplicate leakage between
folds), which is stricter than the random splits most published numbers use — so a
slightly lower number here can still be the more honest one.

## 2. Recovery test — does it know real AMPs from noise?
Scored 12 well-characterized AMPs against 500 random peptides.
- **Recovery:** 100% of known AMPs scored ≥ 0.5 (active).
- **Separation:** ROC-AUC **0.997** distinguishing known AMPs from random decoys
  (decoy mean amp_prob 0.16).

> Honest caveat: several of these classic AMPs (e.g. Magainin-2, Melittin, LL-37) are in
> GRAMPA training data, so this is a **sanity check that the model behaves correctly**, not
> a generalization claim. The cluster-aware CV in §1 is the generalization number.

![recovery](figures/eval_recovery.png)

| Known AMP | amp_prob |
|---|---:|
| Magainin-2 | 0.98 |
| Melittin | 0.99 |
| LL-37 | 0.97 |
| Indolicidin | 0.99 |
| Cecropin-A | 0.99 |
| Protegrin-1 | 0.99 |
| Buforin-II | 0.94 |
| Aurein-1.2 | 0.72 |
| Pexiganan | 1.00 |
| Pleurocidin | 0.99 |
| HNP-1 | 0.92 |
| Dermaseptin | 0.98 |

## 3. Calibration — is `amp_prob` a trustworthy confidence?
Held-out, cluster-aware: isotonic-calibrated **HistGB**, **Brier score 0.070**
(lower is better; 0.25 = uninformative). Points near the diagonal = well-calibrated.

![calibration](figures/eval_calibration.png)

## 4. Applicability domain — are the candidates in-distribution?
Each candidate's distance to its nearest training peptide (standardized features).
**100%** of the shortlist lies within the training distribution
(≤ the 99th percentile of training-to-training distances). Candidates beyond it are
extrapolations and should be treated with extra caution.

![applicability](figures/eval_applicability.png)

## 5. Uncertainty
Every shortlisted candidate carries an `amp_uncertainty` (spread across the calibrated
CV ensemble) in `shortlist.csv` — a per-candidate confidence band on `amp_prob`.

## 6. Limitations (read this)
- **Predictions ≠ proof.** No software confirms a real bacterial kill or human-cell
  safety. Physical MIC/hemolysis validation is Stage 3 (wet lab), out of scope.
- **Negatives are a modeling choice.** Activity negatives are UniProt decoys (answers
  "is this an AMP at all?"), not low-potency peptides. Different negatives → different model.
- **Distribution shift.** The generator can propose peptides unlike anything in training;
  the applicability-domain flag (§4) is a guard, not a guarantee.
- **MIC regression is weak** (R² modest); treat `pred_mic_uM` as a coarse ranking aid,
  not a quantitative prediction.
- **Hemolysis model is the weakest link** (ROC-AUC ≈0.76); simple composition features
  under-capture toxicity. ESM-2 embeddings are the documented upgrade path.
- **Generator posterior collapse.** The VAE's KL term collapsed (decoder ignored the
  latent), so some raw samples echoed training motifs — the novelty filter removes these,
  but a stronger generator (β-tuning / free-bits / fine-tuned protein LM) would improve diversity.

> Every number here is a computational prediction with uncertainty. The value of Amphion
> is a *prioritized, transparent shortlist* for experimental testing — not validated drugs.
