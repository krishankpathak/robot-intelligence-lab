"""
Automated Unit Tests for Cleaning, Transformations, and Leakage-Safe Data Splitting
"""
import pytest
import pandas as pd
import numpy as np

from src.data.dataset_loader import generate_sample_dataset
from src.preprocessing.cleaning import clean_telemetry_data
from src.preprocessing.transformations import build_preprocessor
from src.preprocessing.data_splitting import split_telemetry_data
from src.analysis.feature_engineering import compute_engineered_features


@pytest.fixture
def telemetry_df():
    df = generate_sample_dataset(n_samples=300, seed=42)
    # Inject a couple of test conditions for grouped splitting
    df["test_condition"] = np.repeat([f"run_{i}" for i in range(6)], 50)
    return df


def test_cleaning_deduplication_and_glitches(telemetry_df):
    """Verifies that duplicate rows are removed and negative glitches treated."""
    # Inject duplicate and negative glitch
    corrupted = pd.concat([telemetry_df, telemetry_df.iloc[:5]], ignore_index=True)
    corrupted.loc[10, "position"] = -32000.0

    cleaned, rep = clean_telemetry_data(corrupted, handle_glitches=True)
    assert rep["duplicates_removed"] == 5
    assert rep["glitches_clipped"] >= 1
    assert (cleaned["position"] >= 0).all()


def test_grouped_data_splitting_no_leakage(telemetry_df):
    """
    Verifies that Grouped splitting holds out entire runs so that NO run appears
    in both training and test partitions.
    """
    split_res = split_telemetry_data(telemetry_df, strategy="grouped", test_size=0.33, random_state=42)
    train_df = split_res["train_df"]
    test_df = split_res["test_df"]

    train_runs = set(train_df["test_condition"].unique())
    test_runs = set(test_df["test_condition"].unique())

    # Critical Leakage Guardrail: Disjoint run sets
    intersection = train_runs.intersection(test_runs)
    assert len(intersection) == 0, f"Data leakage detected! Runs in both train and test: {intersection}"
    assert len(train_df) + len(test_df) == len(telemetry_df)


def test_temporal_data_splitting(telemetry_df):
    """Verifies that temporal splitting partitions earlier vs later runs."""
    split_res = split_telemetry_data(telemetry_df, strategy="temporal", test_size=0.33)
    train_runs = split_res["train_runs"]
    test_runs = split_res["test_runs"]

    assert len(set(train_runs).intersection(set(test_runs))) == 0
    # Earliest runs must be in train
    assert "run_0" in train_runs
    assert "run_5" in test_runs


def test_stratified_data_splitting_warning(telemetry_df):
    """Verifies that random stratified split returns an explicit leakage warning."""
    split_res = split_telemetry_data(telemetry_df, strategy="stratified", test_size=0.25, random_state=42)
    assert split_res["leakage_warning"] is not None
    assert "temporal leakage" in split_res["leakage_warning"]


def test_preprocessor_fitting_isolation(telemetry_df):
    """
    Verifies that preprocessor is fitted strictly on training data and transforms test data
    without modifying train statistics.
    """
    features = ["position", "temperature", "voltage"]
    split_res = split_telemetry_data(telemetry_df, strategy="grouped", test_size=0.33, random_state=42)
    X_train = split_res["train_df"][features]
    X_test = split_res["test_df"][features]

    prep = build_preprocessor(numeric_features=features, scaler_type="standard")
    prep.fit(X_train)

    train_transformed = prep.transform(X_train)
    test_transformed = prep.transform(X_test)

    # Train mean should be close to 0 under standard scaling
    assert np.allclose(train_transformed.mean(axis=0), 0.0, atol=1e-2)
    assert train_transformed.shape == (len(X_train), 3)
    assert test_transformed.shape == (len(X_test), 3)


def test_feature_engineering_within_boundaries(telemetry_df):
    """Verifies that rolling and difference features are computed without shape corruption."""
    enriched, new_feats = compute_engineered_features(telemetry_df, window_size=5)
    assert len(enriched) == len(telemetry_df)
    assert "temp_rate_of_change" in new_feats
    assert "voltage_rolling_std_5" in new_feats
    # Check that no NaNs remain
    for f in new_feats:
        assert not enriched[f].isnull().any()
