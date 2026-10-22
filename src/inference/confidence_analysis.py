"""
Confidence and Uncertainty Analysis for ROBOT INTELLIGENCE LAB
Evaluates decision margin, entropy, and prediction uncertainty.
"""
from typing import Any
import numpy as np


def analyze_prediction_confidence(probability: float, threshold: float = 0.5) -> dict[str, Any]:
    """
    Computes decision margin, entropy, and reliability rating.
    """
    prob_fault = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    prob_normal = 1.0 - prob_fault

    # Shannon Entropy (in bits)
    entropy = - (prob_fault * np.log2(prob_fault) + prob_normal * np.log2(prob_normal))

    # Distance from decision boundary
    margin = abs(prob_fault - threshold)

    if margin < 0.1:
        confidence_level = "LOW (Borderline Case)"
    elif margin < 0.3:
        confidence_level = "MODERATE"
    else:
        confidence_level = "HIGH"

    return {
        "probability_fault": np.round(prob_fault, 4),
        "decision_margin": np.round(margin, 4),
        "entropy_bits": np.round(entropy, 4),
        "confidence_level": confidence_level,
    }
