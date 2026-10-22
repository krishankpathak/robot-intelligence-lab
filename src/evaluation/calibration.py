"""
Probability Calibration and Reliability Analysis for ROBOT INTELLIGENCE LAB
"""
from typing import Any
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def evaluate_probability_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> dict[str, Any]:
    """
    Computes calibration curve coordinates and Brier score loss.
    """
    p1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    brier = float(brier_score_loss(y_true, p1))

    prob_true, prob_pred = calibration_curve(y_true, p1, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": float(np.round(brier, 4)),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "interpretation": (
            f"Brier score is {brier:.4f} (lower is better, 0.0 is perfect). "
            "Examine whether predicted probabilities align with observed empirical frequencies."
        ),
    }
