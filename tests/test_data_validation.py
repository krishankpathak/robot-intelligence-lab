"""
Automated Unit Tests for Dataset Loading, Schema Validation, and Quality Audit
"""
import pytest
import pandas as pd
import numpy as np

from src.data.dataset_loader import generate_sample_dataset, load_sample_dataset
from src.data.data_validation import validate_telemetry_schema, validate_inference_input
from src.data.data_quality import run_full_quality_audit, analyze_class_imbalance


@pytest.fixture
def sample_telemetry_df():
    return generate_sample_dataset(n_samples=200, seed=123)


def test_sample_dataset_generation_and_schema(sample_telemetry_df):
    """Verifies that synthetic benchmark generator yields correct schema and origin tag."""
    assert len(sample_telemetry_df) == 200
    expected_cols = ["time", "position", "temperature", "voltage", "label", "motor", "data_origin"]
    for col in expected_cols:
        assert col in sample_telemetry_df.columns
    assert (sample_telemetry_df["data_origin"] == "SYNTHETIC_BENCHMARK_LOCAL").all()


def test_schema_validation_success(sample_telemetry_df):
    """Tests that a valid telemetry DataFrame passes schema validation."""
    result = validate_telemetry_schema(
        sample_telemetry_df,
        required_features=["position", "temperature", "voltage"],
        require_target=True,
        target_col="label",
    )
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert result["schema_info"]["num_rows"] == 200


def test_schema_validation_missing_feature(sample_telemetry_df):
    """Verifies that missing required sensor features produce descriptive validation errors."""
    corrupted_df = sample_telemetry_df.drop(columns=["voltage"])
    result = validate_telemetry_schema(
        corrupted_df,
        required_features=["position", "temperature", "voltage"],
        require_target=True,
    )
    assert result["is_valid"] is False
    assert any("voltage" in err for err in result["errors"])


def test_schema_validation_missing_target(sample_telemetry_df):
    """Verifies that missing target produces descriptive error when require_target=True."""
    no_target_df = sample_telemetry_df.drop(columns=["label"])
    result = validate_telemetry_schema(no_target_df, require_target=True, target_col="label")
    assert result["is_valid"] is False
    assert any("Missing target column" in err for err in result["errors"])


def test_data_quality_audit(sample_telemetry_df):
    """Verifies quality audit reports duplicate rows, missing values, and class imbalance."""
    report = run_full_quality_audit(sample_telemetry_df, target_col="label")
    assert report["total_records"] == 200
    assert report["duplicate_rows"] == 0
    assert "imbalance_report" in report
    assert report["imbalance_report"]["has_target"] is True
    assert 0 in report["imbalance_report"]["counts"]
    assert 1 in report["imbalance_report"]["counts"]


def test_validate_inference_input_valid():
    """Tests inference input validation for single dictionary."""
    valid_sample = {"position": 450.0, "temperature": 35.5, "voltage": 7210.0}
    is_valid, err, mat = validate_inference_input(valid_sample, ["position", "temperature", "voltage"])
    assert is_valid is True
    assert err == ""
    assert mat.shape == (1, 3)


def test_validate_inference_input_missing_feature():
    """Tests inference input validation fails with missing feature."""
    invalid_sample = {"position": 450.0, "temperature": 35.5}
    is_valid, err, mat = validate_inference_input(invalid_sample, ["position", "temperature", "voltage"])
    assert is_valid is False
    assert "voltage" in err
    assert mat is None
