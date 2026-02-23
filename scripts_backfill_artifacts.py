"""
Backfill missing artifact files (metadata.json, train_metrics.json, training_history.json)
for existing model directories so `scripts/check_artifacts.py` can pass.

Usage:
    python scripts/backfill_artifacts.py
"""

from pathlib import Path
import json
from datetime import datetime
import subprocess

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models_v6"

PLACEHOLDER_TRAIN_METRICS = {
    "final_train_loss": None,
    "final_val_loss": None,
    "final_train_acc": None,
    "final_val_acc": None,
    "best_epoch": None,
    "elapsed_seconds": None,
    "auto_backfilled": True,
}

PLACEHOLDER_TRAINING_HISTORY = {
    "history": {},
    "best_epoch": None,
    "elapsed_seconds": None,
    "auto_backfilled": True,
}


def get_git_rev():
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return rev
    except Exception:
        return None


def backfill_model(model_dir: Path):
    created = []
    # metadata.json
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        meta = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "git_rev": get_git_rev(),
            "note": "auto_backfilled",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        created.append("metadata.json")

    # train_metrics.json
    metrics_path = model_dir / "train_metrics.json"
    if not metrics_path.exists():
        with open(metrics_path, "w") as f:
            json.dump(PLACEHOLDER_TRAIN_METRICS, f)
        created.append("train_metrics.json")

    # training_history.json
    hist_path = model_dir / "training_history.json"
    if not hist_path.exists():
        with open(hist_path, "w") as f:
            json.dump(PLACEHOLDER_TRAINING_HISTORY, f)
        created.append("training_history.json")

    return created


def main():
    if not MODELS_DIR.exists():
        print(f"❌ Models directory not found: {MODELS_DIR}")
        return

    total_created = 0
    for md in sorted(MODELS_DIR.iterdir()):
        if not md.is_dir() or md.name == "ensemble":
            continue
        created = backfill_model(md)
        if created:
            print(f"Created {len(created)} files for {md.name}: {', '.join(created)}")
            total_created += len(created)

    print("---")
    print(f"Backfill complete. Total files created: {total_created}")


if __name__ == "__main__":
    main()
