"""
Automated Unit Tests for Local Inference Service, Artifact Storage, and Integrity Verification
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.dataset_loader import generate_sample_dataset
from src.preprocessing.transformations import build_preprocessor
from src.models.decision_tree_model import create_decision_tree_model
from src.models.model_registry import save_model_artifact, load_model_artifact
from src.inference.prediction_service import LocalPredictionService
from sklearn.pipeline import Pipeline


@pytest.fixture
def trained_model_artifact(tmp_path):
    """Trains a quick pipeline and saves it to a temporary directory."""
    df = generate_sample_dataset(n_samples=200, seed=42)
    features = ["position", "temperature", "voltage"]

    prep = build_preprocessor(numeric_features=features, scaler_type="standard")
    clf = create_decision_tree_model(max_depth=4)
    pipe = Pipeline([("prep", prep), ("clf", clf)])

    pipe.fit(df[features], df["label"])

    artifact_file = save_model_artifact(
        pipeline=pipe,
        model_name="Test Decision Tree",
        feature_names=features,
        metrics={"accuracy": 0.95, "f1": 0.88},
        save_dir=tmp_path,
    )
    return artifact_file, tmp_path


def test_save_and_load_artifact_integrity(trained_model_artifact):
    """Verifies that an artifact can be reloaded and checksum integrity verified."""
    artifact_file, _ = trained_model_artifact
    loaded_pipe, meta = load_model_artifact(artifact_file)

    assert loaded_pipe is not None
    assert meta["model_name"] == "Test Decision Tree"
    assert meta["feature_names"] == ["position", "temperature", "voltage"]
    assert "sha256_checksum" in meta


def test_artifact_tamper_detection(trained_model_artifact):
    """Verifies that tampering with a saved joblib file triggers a security checksum error."""
    artifact_file, _ = trained_model_artifact

    # Tamper with file bytes
    with open(artifact_file, "ab") as f:
        f.write(b"TAMPER_DATA")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_artifact(artifact_file)


def test_local_prediction_service_single_and_batch(trained_model_artifact):
    """Verifies single-sample and batch local inference using the PredictionService."""
    artifact_file, _ = trained_model_artifact

    baseline_stats = {
        "position": {"mean": 500.0, "std": 200.0, "min": 0.0, "max": 999.0},
        "temperature": {"mean": 35.0, "std": 5.0, "min": 20.0, "max": 65.0},
        "voltage": {"mean": 7200.0, "std": 100.0, "min": 6800.0, "max": 7600.0},
    }

    service = LocalPredictionService(
        model_artifact_path=artifact_file,
        training_baseline_stats=baseline_stats,
        decision_threshold=0.5,
    )

    # 1. Single sample prediction
    sample = {"position": 510.0, "temperature": 34.0, "voltage": 7210.0}
    res = service.predict_sample(sample)

    assert res["success"] is True
    assert res["predicted_class"] in [0, 1]
    assert "explanation" in res
    assert "explanation_text" in res["explanation"]

    # 2. Batch prediction
    batch_df = pd.DataFrame([
        {"position": 500.0, "temperature": 33.0, "voltage": 7220.0},
        {"position": 850.0, "temperature": 55.0, "voltage": 6850.0},
    ])
    out_df = service.predict_batch(batch_df)

    assert len(out_df) == 2
    assert "predicted_label" in out_df.columns
    assert set(out_df["predicted_label"]).issubset({0, 1})
