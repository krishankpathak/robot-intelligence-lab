"""
Automated Unit Tests for Candidate ML Models (Baseline, Linear, Trees, Ensembles)
"""
import pytest
import pandas as pd
import numpy as np

from src.data.dataset_loader import generate_sample_dataset
from src.preprocessing.data_splitting import split_telemetry_data
from src.models.model_training import train_single_model, train_all_models, SUPPORTED_MODEL_TYPES


@pytest.fixture
def dataset_partitions():
    df = generate_sample_dataset(n_samples=400, seed=99)
    df["test_condition"] = np.repeat([f"run_{i}" for i in range(8)], 50)
    split_res = split_telemetry_data(df, strategy="grouped", test_size=0.25, random_state=42)
    return split_res["train_df"], split_res["test_df"]


@pytest.mark.parametrize("model_name", SUPPORTED_MODEL_TYPES)
def test_each_candidate_model_trains_and_predicts(dataset_partitions, model_name):
    """Verifies that all standard candidate models train, produce valid predictions, and return metrics."""
    train_df, test_df = dataset_partitions
    features = ["position", "temperature", "voltage"]

    res = train_single_model(
        model_name=model_name,
        X_train=train_df[features],
        y_train=train_df["label"].to_numpy(),
        X_test=test_df[features],
        y_test=test_df["label"].to_numpy(),
        feature_names=features,
    )

    pipeline = res["pipeline"]
    metrics = res["metrics"]

    # Test pipeline predictions
    y_pred = pipeline.predict(test_df[features])
    assert len(y_pred) == len(test_df)
    assert set(np.unique(y_pred)).issubset({0, 1})

    # Verify metric keys
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "fit_time_sec" in metrics
    assert metrics["fit_time_sec"] >= 0.0

    # If model supports probabilities
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(test_df[features])
        assert probs.shape == (len(test_df), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


def test_train_all_models_orchestrator(dataset_partitions):
    """Verifies that train_all_models executes all models and returns dictionary."""
    train_df, test_df = dataset_partitions
    all_res = train_all_models(train_df, test_df, feature_names=["position", "temperature", "voltage"])
    assert len(all_res) == len(SUPPORTED_MODEL_TYPES)
    for name in SUPPORTED_MODEL_TYPES:
        assert name in all_res
        assert "pipeline" in all_res[name]
        assert "metrics" in all_res[name]
