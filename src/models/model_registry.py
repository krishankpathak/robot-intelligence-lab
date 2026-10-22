"""
Model Registry and Artifact Storage for ROBOT INTELLIGENCE LAB
Saves and loads fitted preprocessor-model pipelines alongside cryptographic checksums and training provenance.
"""
from typing import Any
from pathlib import Path
import hashlib
import json
import joblib

from config.settings import ARTIFACTS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("model_registry")


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes SHA-256 hash of a file for integrity and provenance verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def save_model_artifact(
    pipeline,
    model_name: str,
    feature_names: list[str],
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    save_dir: Path | str = ARTIFACTS_DIR,
) -> Path:
    """
    Saves the entire scikit-learn Pipeline (preprocessing + model) and metadata to disk.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    sanitized_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_file = save_path / f"{sanitized_name}_pipeline.joblib"
    meta_file = save_path / f"{sanitized_name}_metadata.json"

    # Save pipeline
    joblib.dump(pipeline, model_file)
    checksum = compute_file_sha256(model_file)

    # Save metadata
    full_meta = {
        "model_name": model_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "sha256_checksum": checksum,
        "extra_metadata": metadata or {},
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    logger.info(f"Saved model artifact to {model_file} (SHA256: {checksum[:12]}...)")
    return model_file


def load_model_artifact(model_file: Path | str) -> tuple[Any, dict[str, Any]]:
    """
    Loads a saved pipeline and its corresponding metadata, verifying file integrity.
    """
    file_path = Path(model_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {file_path}")

    meta_path = file_path.with_name(file_path.stem.replace("_pipeline", "") + "_metadata.json")
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_hash = metadata.get("sha256_checksum")
        if expected_hash:
            actual_hash = compute_file_sha256(file_path)
            if actual_hash != expected_hash:
                raise ValueError("Security/Integrity Error: Model artifact checksum mismatch! Artifact may be tampered.")

    pipeline = joblib.load(file_path)
    logger.info(f"Loaded verified model artifact from {file_path}")
    return pipeline, metadata


def list_saved_artifacts(artifacts_dir: Path | str = ARTIFACTS_DIR) -> list[dict[str, Any]]:
    """Lists all saved models and their metadata in the artifacts directory."""
    p = Path(artifacts_dir)
    if not p.exists():
        return []

    saved = []
    for f in p.glob("*_pipeline.joblib"):
        meta_f = p / (f.stem.replace("_pipeline", "") + "_metadata.json")
        meta = {}
        if meta_f.exists():
            try:
                with open(meta_f, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
            except Exception:
                pass
        saved.append({
            "model_path": str(f),
            "model_name": meta.get("model_name", f.stem),
            "features": meta.get("feature_names", []),
            "f1_score": meta.get("metrics", {}).get("f1"),
            "recall": meta.get("metrics", {}).get("recall"),
        })
    return saved
