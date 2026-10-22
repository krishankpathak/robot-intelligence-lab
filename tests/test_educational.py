"""
Automated Unit Tests for From-Scratch Educational Algorithms and Custom Metrics
"""
import pytest
import numpy as np

from src.models.educational_scratch import EducationalLogisticRegression, EducationalDecisionStump
from src.evaluation.scratch_metrics import (
    compute_scratch_confusion_matrix,
    compute_scratch_metrics,
    verify_against_sklearn,
)


def test_scratch_sigmoid_numerical_stability():
    """Tests that sigmoid handles extreme positive and negative inputs without overflow or NaN."""
    extreme_inputs = np.array([-1000.0, -500.0, -10.0, 0.0, 10.0, 500.0, 1000.0])
    sig_vals = EducationalLogisticRegression._sigmoid(extreme_inputs)

    assert not np.isnan(sig_vals).any()
    assert (sig_vals >= 0.0).all() and (sig_vals <= 1.0).all()
    assert np.isclose(sig_vals[3], 0.5)  # sigmoid(0) == 0.5
    assert np.isclose(sig_vals[0], 0.0, atol=1e-5)
    assert np.isclose(sig_vals[-1], 1.0, atol=1e-5)


def test_scratch_logistic_regression_loss_decreases():
    """Verifies that gradient descent decreases cross-entropy loss on a controlled dataset."""
    rng = np.random.default_rng(42)
    # Generate 100 separable points
    X0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(50, 2))
    X1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = EducationalLogisticRegression(learning_rate=0.2, n_iterations=100, random_state=42)
    model.fit(X, y)

    # Loss must be lower at end of training than initial step
    initial_loss = model.loss_history[0]
    final_loss = model.loss_history[-1]
    assert final_loss < initial_loss
    assert final_loss < 0.2

    # Predict
    preds = model.predict(X)
    assert len(preds) == 100
    acc = np.mean(preds == y)
    assert acc >= 0.95


def test_scratch_decision_stump_gini_and_split():
    """Verifies Gini calculation and optimal split detection in Decision Stump."""
    # Pure set
    assert EducationalDecisionStump.compute_gini(np.array([1, 1, 1])) == 0.0
    assert EducationalDecisionStump.compute_gini(np.array([0, 0, 0])) == 0.0
    # Equal split Gini: 1 - (0.5^2 + 0.5^2) = 0.5
    assert np.isclose(EducationalDecisionStump.compute_gini(np.array([0, 1])), 0.5)

    # Simple 1D separable dataset: X <= 5 is class 0, X > 5 is class 1
    X = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    stump = EducationalDecisionStump()
    stump.fit(X, y)

    assert stump.gini_gain > 0.4
    assert 3.0 < stump.best_threshold < 7.0
    preds = stump.predict(X)
    assert (preds == y).all()


def test_scratch_metrics_zero_division():
    """Verifies zero-division safety when there are 0 positive predictions or 0 actual positives."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = compute_scratch_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_scratch_metrics_agreement_with_sklearn():
    """Verifies custom metrics agree exactly with scikit-learn on identical predictions."""
    rng = np.random.default_rng(123)
    y_true = rng.choice([0, 1], size=150, p=[0.85, 0.15])
    y_pred = rng.choice([0, 1], size=150, p=[0.80, 0.20])

    comparison = verify_against_sklearn(y_true, y_pred, tolerance=1e-4)
    assert comparison["all_match"] is True, f"Mismatch found: {comparison['checks']}"
