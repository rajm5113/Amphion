# Model card — MIC regressor (log10 MIC, uM)

**Task:** predict a peptide's best potency `log10(min MIC, uM)` — lower = more potent.
**Data:** GRAMPA positives (6,448 peptides, 29 features).
**Selected model:** RandomForest (lowest cluster-aware RMSE).
**CV:** GroupKFold on homology cluster_id (no near-duplicate leakage).

| Model | RMSE | MAE | R² | Spearman |
|---|---:|---:|---:|---:|
| ElasticNet | 0.791 | 0.611 | 0.186 | 0.457 |
| RandomForest | 0.755 | 0.570 | 0.258 | 0.518 |
| HistGB | 0.763 | 0.577 | 0.244 | 0.504 |

Predictions feed the pipeline's potency term (Phase 6 ranking) and the
`pred_mic_uM` field of `score_activity()`.

> Computational predictions with uncertainty, not validated MIC measurements.
