"""Export the processed datasets as small CSVs for the ESM-2 Kaggle notebook.

These let the notebook train on the EXACT same sequences, labels, and homology
clusters as the 29-feature baseline — so the ESM-vs-baseline comparison is honest
and the resulting models drop straight into the pipeline.

Run:  .venv/Scripts/python scripts/export_esm_inputs.py
Then upload the data/external/esm_inputs/ folder to Kaggle as a Dataset.
"""

from __future__ import annotations

import pandas as pd

from amphion import get_config, get_logger
from amphion.utils import ensure_dir

log = get_logger("amphion.export_esm")


def main():
    cfg = get_config()
    proc = cfg.resolve_path("processed")
    out = ensure_dir(cfg.resolve_path("external") / "esm_inputs")

    pd.read_parquet(proc / "activity_classification.parquet")[
        ["sequence", "label", "cluster_id", "in_balanced"]
    ].to_csv(out / "activity.csv", index=False)

    pd.read_parquet(proc / "activity_regression.parquet")[
        ["sequence", "min_log_mic_uM", "cluster_id"]
    ].to_csv(out / "mic.csv", index=False)

    pd.read_parquet(proc / "hemolysis.parquet")[
        ["sequence", "hemolytic", "cluster_id"]
    ].to_csv(out / "hemolysis.csv", index=False)

    for f in ["activity.csv", "mic.csv", "hemolysis.csv"]:
        n = sum(1 for _ in open(out / f)) - 1
        log.info("wrote %s (%d rows)", out / f, n)
    print(f"\nUpload this folder to Kaggle as a Dataset:\n  {out}")


if __name__ == "__main__":
    main()
