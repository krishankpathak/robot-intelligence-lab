"""
Deterministic Local Prediction Explanation Module for ROBOT INTELLIGENCE LAB
Generates clear, rule-based, template-driven explanations for individual sensor observations
without external LLMs or third-party cloud APIs.
"""
from typing import Any
import pandas as pd
import numpy as np


def explain_single_prediction(
    sample_features: dict[str, float],
    predicted_class: int,
    predicted_prob: float | None,
    training_baseline_stats: dict[str, dict[str, float]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Produces a local explanation for a single prediction by comparing its sensor readings
    with empirical training data baselines (mean, std, min, max).
    """
    deviations = []
    ood_flags = []

    for feat, val in sample_features.items():
        if feat in training_baseline_stats:
            stats = training_baseline_stats[feat]
            mean_val = stats.get("mean", 0.0)
            std_val = max(stats.get("std", 1.0), 1e-4)
            min_val = stats.get("min", -np.inf)
            max_val = stats.get("max", np.inf)

            z_score = (val - mean_val) / std_val

            # Check out of distribution (OOD)
            is_ood = (val < min_val) or (val > max_val)
            if is_ood:
                ood_flags.append(
                    f"Sensor '{feat}' value ({val:.2f}) is OUTSIDE training range [{min_val:.1f}, {max_val:.1f}]."
                )

            deviations.append({
                "feature": feat,
                "value": np.round(val, 2),
                "baseline_mean": np.round(mean_val, 2),
                "z_score": np.round(z_score, 2),
                "is_abnormal": abs(z_score) > 2.0,
            })

    # Sort deviations by absolute z-score
    deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    top_deviations = deviations[:3]

    # Generate deterministic text explanation
    class_label_str = "FAULT DETECTED (Class 1)" if predicted_class == 1 else "NORMAL OPERATION (Class 0)"

    prob_str = f"with estimated probability {predicted_prob*100:.1f}%" if predicted_prob is not None else ""
    summary_lines = [
        f"Model Classification: {class_label_str} {prob_str} (Decision Threshold: {threshold:.2f})."
    ]

    if predicted_class == 1:
        influential = [f"'{d['feature']}' ({d['value']:.1f}, {d['z_score']:+.1f}σ from mean)" for d in top_deviations if d["is_abnormal"]]
        if influential:
            summary_lines.append(
                f"Primary telemetry drivers triggering fault classification: {', '.join(influential)}."
            )
        else:
            summary_lines.append(
                "Prediction was driven by subtle multivariate interactions across multiple sensor readings."
            )
    else:
        summary_lines.append(
            "All sensor readings reside within expected operating tolerance limits."
        )

    if ood_flags:
        summary_lines.append("CAUTION: Out-of-Distribution Warning detected: " + " ".join(ood_flags))

    return {
        "predicted_class": predicted_class,
        "predicted_prob": predicted_prob,
        "class_label": class_label_str,
        "deviations": deviations,
        "top_deviations": top_deviations,
        "is_out_of_distribution": len(ood_flags) > 0,
        "ood_warnings": ood_flags,
        "explanation_text": "\n".join(summary_lines),
        "caveat": (
            "Deterministic diagnostic note: This explanation summarizes which sensors departed most significantly "
            "from normal baseline distributions. It reflects mathematical model attribution, not verified physical disassembly."
        ),
    }
