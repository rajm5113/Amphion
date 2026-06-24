# Methods

A concise technical writeup of the Amphion Stage-1 in-silico pipeline. Companion to the
portfolio [`README.md`](../README.md) and the evaluation in [`reports/evaluation.md`](../reports/evaluation.md).

## 1. Data

| Source | Role | Processing |
|---|---|---|
| **GRAMPA** (consolidated DBAASP/DRAMP/YADAMP/APD/DADP) | AMP positives + MIC | 51,345 MIC rows → 6,448 unique canonical peptides; potency = `min(log₁₀ MIC µM)` across strains |
| **AMPlify** UniProt-derived non-AMPs | non-AMP negatives | 3,815 cleaned, leakage-removed |
| **HemoPI2** (Raghava lab) | hemolysis labels + HC50 | 1,926 peptides; label = HC50 ≤ 100 µM (reproduces source labels 100%) |

**Cleaning (uniform):** uppercase; keep only the 20 standard amino acids; length 5–50; dedupe.
**Leakage control:** any sequence present in both AMP and non-AMP sets is removed.

## 2. Homology-aware splitting

Near-identical sequences split across train/test inflate scores. Every sequence is assigned a
`cluster_id` via greedy single-linkage clustering (CD-HIT-like, rapidfuzz identity ≥ 0.6).
All cross-validation is **cluster-aware** (`StratifiedGroupKFold` / `GroupKFold` on `cluster_id`),
so near-duplicates never straddle a fold. Reported metrics are therefore lower — and more
honest — than the random-split numbers common in the literature.

## 3. Features

29 interpretable, biology-motivated features per peptide:
- **Amino-acid composition** — 20 fractions.
- **Physicochemical** — length, net charge, charge density, mean Kyte–Doolittle hydrophobicity,
  fraction hydrophobic / polar / positive / negative, aromaticity.

Deliberately simple and explainable; ESM-2 embeddings are the documented accuracy-upgrade path.

## 4. Models

**Activity classifier.** LogReg / RandomForest / HistGradientBoosting compared under both random
and cluster-aware CV; best by cluster-aware AUC; isotonic-calibrated; saved with a model card.
Cluster-aware ROC-AUC ≈ 0.95 (random-split ≈ 0.97). The biological sanity check — charge-related
features dominating importance — is asserted in the card.

**MIC regressor.** ElasticNet / RF / HistGB on `log₁₀(min MIC)`; cluster-aware `GroupKFold`;
RMSE ≈ 0.76, Spearman ≈ 0.50. **Deployability-aware selection** picks the most compact model within
3% of the best RMSE — gradient boosting (~0.4 MB) over a 194 MB Random Forest for a <2% accuracy gain.
Used as a coarse potency signal, not a quantitative predictor.

**Toxicity classifier.** Class-weighted LogReg / RF / HistGB on the hemolysis set; selected by
cluster-aware **PR-AUC** (imbalance-aware); isotonic-calibrated.

Unified API: `score_activity(seq) → {amp_prob, pred_log_mic, pred_mic_uM}`,
`score_toxicity(seq) → {hemolytic_prob}`.

**ESM-2 hybrid (deployed).** ESM-2 (150M) embeddings were benchmarked against the 29 hand-crafted
features on identical splits. ESM-2 *improved* **activity** (AUC 0.954 → 0.966) and **toxicity**
(PR-AUC 0.704 → 0.720) but *lost* on **MIC** (RMSE 0.755 → 0.818 — potency is charge-driven, encoded
directly by the hand-crafted features and diluted by mean-pooled embeddings). Amphion therefore
deploys a **hybrid**: ESM-2 for activity + toxicity, hand-crafted features for MIC. The scorer
auto-selects ESM models when present, else the CPU-only baseline.

## 5. Generator

A character-level **sequence VAE** over the 23-token vocabulary (PAD/SOS/EOS + 20 AAs):
GRU encoder → latent *z* (dim 32) → GRU decoder (autoregressive, teacher-forced). Loss =
reconstruction cross-entropy + β·KL with KL annealing. Trained on AMP positives on a free Kaggle
GPU (~36 s, 60 epochs). Sampling draws *z* ∼ 𝒩(0, I) and decodes with temperature.

**Validity gate:** canonical (guaranteed by vocab), length 5–50, not an exact training duplicate,
unique. 5,000 valid candidates produced; generated K+R fraction (0.244) ≈ training (0.256),
confirming the model learned AMP-like cationicity.

**Known issue — posterior collapse:** the KL term collapsed to ≈0 (decoder learned to ignore *z*),
so the VAE behaves partly like an autoregressive LM and some raw samples echo training motifs.
The novelty filter removes these; a stronger generator (β-tuning, free-bits, or a fine-tuned
protein LM) is the upgrade path.

## 6. Novelty filter

For each candidate, the maximum sequence identity (rapidfuzz) against the **union of all known
sequences** (positives + negatives + hemolysis set). Novel iff max identity < 0.6.
`novelty_score = 1 − max_identity`. Of 5,000 candidates, 1,321 (26%) were novel.

## 7. Composite ranking

Survivors of the gates (active, safe, novel) are ranked by a config-weighted score:

```
rank_score = 0.30·amp_prob
           + 0.30·potency_term      # min-max normalized −pred_log_mic across the pool
           + 0.25·(1 − hemolytic_prob)
           + 0.10·novelty_score
           + 0.05·synthesizability  # penalizes extreme length / charge / cysteine / hydrophobicity
```

Gates: `amp_prob ≥ 0.5`, `hemolytic_prob ≤ 0.5`, `novel = true`. Result: **467 / 5,000** shortlisted.
Each candidate records every sub-score plus an `amp_uncertainty` (spread across the calibrated CV
ensemble) for full provenance.

## 8. Evaluation

- **Recovery:** 100% of 12 classic AMPs scored active; ROC-AUC 0.997 vs random decoys (sanity
  check — several are in training, so not a generalization claim).
- **Calibration:** held-out cluster-aware reliability diagram, Brier 0.070.
- **Applicability domain:** 100% of the shortlist lies within the 99th-percentile nearest-neighbour
  distance of the training distribution (standardized feature space).
- **Uncertainty:** per-candidate ensemble spread on `amp_prob`.

## 9. Limitations

Predictions ≠ proof; no wet-lab validation; MIC and hemolysis models are weak (simple features);
generator posterior collapse; activity negatives are a modeling choice (UniProt decoys answer
"is this an AMP at all?", not "which real peptide is most potent?"). See `reports/evaluation.md`.

## References
1. AMP Scanner v2 — Veltri et al., *Bioinformatics* 2018.
2. AMPlify — Li et al., *BMC Genomics* 2022.
3. HemoPI — Chaudhary et al., *Sci. Rep.* 2016.
4. HemoPI2 — Rathore et al., *Commun. Biol.* 2025.
5. GRAMPA — Witten & Witten, 2019.
6. ESM-2 — Lin et al., *Science* 2023.
