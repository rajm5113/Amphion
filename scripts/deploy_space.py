"""Deploy the Gradio demo to a free Hugging Face Space.

Usage:
    HF_TOKEN=<write-token>  [HF_SPACE=user/Amphion]  python scripts/deploy_space.py

- Token: create a *write* token at https://huggingface.co/settings/tokens
- Creates the Space (Gradio SDK) if it doesn't exist, then uploads only what the
  demo needs (package, config, the small hybrid models, processed parquets, app).
- ESM-2 weights (~600 MB) download on the Space at first run; not uploaded here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Only the files the live demo actually loads.
MODELS = ["activity_clf_esm.joblib", "tox_clf_esm.joblib", "mic_reg.joblib", "esm_config.json"]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face *write* token "
                 "(https://huggingface.co/settings/tokens).")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("Install the client first:  .venv/Scripts/python -m pip install huggingface_hub")

    api = HfApi(token=token)
    # Default the Space to <your-username>/Amphion (derived from the token).
    repo_id = os.environ.get("HF_SPACE")
    if not repo_id:
        user = api.whoami(token=token)["name"]
        repo_id = f"{user}/Amphion"
    create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    def up_file(local, remote):
        api.upload_file(repo_id=repo_id, repo_type="space", token=token,
                        path_or_fileobj=str(ROOT / local), path_in_repo=remote)
        print("  uploaded", remote)

    # 1) the amphion package — both top-level files (amphion/*.py) AND sub-packages
    # (amphion/*/*.py). NOTE: huggingface_hub uses fnmatch, where "**" does NOT match
    # zero dirs, so a single "amphion/**/*.py" silently drops __init__.py/config.py/etc.
    api.upload_folder(repo_id=repo_id, repo_type="space", token=token,
                      folder_path=str(ROOT / "src"), path_in_repo="src",
                      allow_patterns=["amphion/*.py", "amphion/*/*.py"])
    print("  uploaded src/amphion")
    # 2) config + small hybrid models + processed parquets (for novelty)
    up_file("config.yaml", "config.yaml")
    for m in MODELS:
        up_file(f"models/{m}", f"models/{m}")
    api.upload_folder(repo_id=repo_id, repo_type="space", token=token,
                      folder_path=str(ROOT / "data/processed"), path_in_repo="data/processed",
                      allow_patterns=["*.parquet"])
    print("  uploaded data/processed/*.parquet")
    # 3) app entry at the Space root (README has the HF metadata)
    up_file("app/app.py", "app.py")
    up_file("app/requirements.txt", "requirements.txt")
    up_file("app/README.md", "README.md")

    print(f"\nDone -> https://huggingface.co/spaces/{repo_id}")
    print("First load builds the env + downloads ESM-2 (~600 MB) — give it a few minutes.")


if __name__ == "__main__":
    main()
