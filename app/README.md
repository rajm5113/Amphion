---
title: Amphion AMP Designer
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: other
---

# Amphion demo (Hugging Face Spaces)

Paste a peptide → predicted antimicrobial probability, potency (MIC), hemolytic
probability, novelty, and the pipeline's pass/fail gate verdict.

## Run locally
```bash
.venv/Scripts/python app/app.py     # from the repo root
```

## Deploy to a free Hugging Face Space
1. Create a new **Gradio** Space.
2. Add these to the Space repo:
   - `app.py` (this folder's `app.py`) and `requirements.txt` (this folder's),
   - the `src/amphion/` package,
   - the committed hybrid models: `models/activity_clf_esm.joblib`, `models/tox_clf_esm.joblib`,
     `models/mic_reg.joblib`, `models/esm_config.json`,
   - `data/processed/*.parquet` (needed for the novelty lookup).
3. The metadata header above tells the Space to run `app.py`.

> **Model note.** The deployed hybrid models are small and **committed** in `models/`, so the
> demo runs on clone. The **ESM-2 weights (~600 MB)** download from the PyTorch hub on first
> run and are cached — the first request is slow, then fast. (Larger baseline fallbacks stay
> git-ignored.) For a snappier Space, swap to `esm2_t12_35M` and re-run the ESM notebook.

> Predictions only — not validated results.
