"""
Feature Importance Analysis for ROBOT INTELLIGENCE LAB
Extracts tree-based Gini importance and model-agnostic held-out Permutation Importance
with explicit caveats regarding correlation and non-causality.
"""
from typing import Any
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance


CAVEAT_EXPLANATION = (
    "Caveat on Feature Importance: High importance indicates strong predictive association, "
    "NOT physical causation. When physical sensors are correlated (e.g. temperature and voltage sag), "
    "importance scores can be shared or split between them."
)


def extract_tree_importance(fitted_model, feature_names: list[str]) -> pd.DataFrame:
    """Extracts MDI (Mean Decrease in Impurity) feature importances from a tree estimator."""
    clf = getattr(fitted_model, "named_steps", {}).get("clf", fitted_model)

    if not hasattr(clf, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})

    importances = clf.feature_importances_
    total = np.sum(importances)
    normalized = (importances / total * 100) if total > 0 else importances

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_raw": importances,
        "importance_percent": np.round(normalized, 2),
    }).sort_values(by="importance_raw", ascending=False).reset_index(drop=True)

    return df_imp


def compute_held_out_permutation_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes permutation importance on held-out test data.
    Measures the drop in F1-score when a feature is randomly shuffled.
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": np.round(result.importances_mean, 4),
        "std_importance": np.round(result.importances_std, 4),
    }).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)

    return df_perm
