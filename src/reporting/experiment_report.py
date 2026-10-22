"""
Experiment Reporting and Tracking Module for ROBOT INTELLIGENCE LAB
Saves structured JSON/CSV experiment logs capturing hyperparameters, data fingerprints, and metrics.
"""
from typing import Any
from pathlib import Path
from datetime import datetime
import json
import hashlib
import platform
import pandas as pd
import numpy as np

from config.settings import REPORTS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("experiment_report")


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Computes a lightweight MD5 hash fingerprint of DataFrame shape and column contents."""
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    h.update(str(list(df.columns)).encode())
    # Hash sample values
    sample_bytes = df.head(50).to_csv(index=False).encode()
    h.update(sample_bytes)
    return h.hexdigest()[:12]


def log_experiment_run(
    model_name: str,
    feature_names: list[str],
    metrics: dict[str, Any],
    split_strategy: str,
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    random_seed: int = 42,
    threshold: float = 0.5,
    artifact_path: str | None = None,
    reports_dir: Path | str = REPORTS_DIR,
) -> Path:
    """Logs a completed training and evaluation run to a structured JSON file."""
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"exp_{model_name.lower().replace(' ', '_')}_{timestamp_str}"
    log_file = rep_path / f"{run_id}.json"

    experiment_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "feature_names": feature_names,
        "split_strategy": split_strategy,
        "train_samples": train_shape[0],
        "test_samples": test_shape[0],
        "random_seed": random_seed,
        "decision_threshold": threshold,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "system_info": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
        },
        "metrics": metrics,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)

    logger.info(f"Logged experiment record to {log_file}")
    return log_file


def load_all_experiment_reports(reports_dir: Path | str = REPORTS_DIR) -> pd.DataFrame:
    """Loads all experiment JSON logs into a unified historical DataFrame."""
    p = Path(reports_dir)
    if not p.exists():
        return pd.DataFrame()

    records = []
    for f in p.glob("exp_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as jf:
                d = json.load(jf)
                m = d.get("metrics", {})
                records.append({
                    "run_id": d.get("run_id"),
                    "timestamp": d.get("timestamp"),
                    "model_name": d.get("model_name"),
                    "split_strategy": d.get("split_strategy"),
                    "train_samples": d.get("train_samples"),
                    "test_samples": d.get("test_samples"),
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1_score": m.get("f1"),
                    "pr_auc": m.get("pr_auc"),
                    "roc_auc": m.get("roc_auc"),
                    "artifact_path": d.get("artifact_path"),
                })
        except Exception as e:
            logger.warning(f"Failed to read report {f}: {e}")

    df_hist = pd.DataFrame(records)
    if not df_hist.empty and "timestamp" in df_hist.columns:
        df_hist = df_hist.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    return df_hist
