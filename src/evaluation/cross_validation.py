"""
Cross-Validation Module with Strict Within-Fold Preprocessing
Guarantees zero data leakage by refitting transformations independently inside each fold.
"""
from typing import Any
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline

from src.preprocessing.transformations import build_preprocessor
from src.evaluation.classification_metrics import evaluate_classifier_predictions
from src.utils.logging_config import setup_logger

logger = setup_logger("cross_validation")


def run_safe_cross_validation(
    estimator_factory: callable,
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "label",
    group_col: str | None = "test_condition",
    n_splits: int = 5,
    cv_type: str = "group",
    scaler_type: str = "standard",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Executes cross-validation with preprocessing pipelines fit exclusively within each fold.
    Supports 'group', 'stratified', or 'standard' k-fold.
    """
    X = df[features]
    y = df[target_col].to_numpy()

    if cv_type == "group" and group_col in df.columns:
        groups = df[group_col].to_numpy()
        unique_groups = np.unique(groups)
        actual_splits = min(n_splits, len(unique_groups))
        cv = GroupKFold(n_splits=actual_splits)
        split_gen = cv.split(X, y, groups=groups)
    elif cv_type == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_gen = cv.split(X, y)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_gen = cv.split(X, y)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_gen, start=1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Check if validation split has at least one class
        if len(np.unique(y_val)) == 0:
            continue

        # Fit preprocessor strictly on X_train
        preprocessor = build_preprocessor(numeric_features=features, scaler_type=scaler_type)
        model = estimator_factory()

        pipe = Pipeline([("prep", preprocessor), ("clf", model)])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val) if hasattr(pipe, "predict_proba") else None

        res = evaluate_classifier_predictions(y_val, y_pred, y_prob)
        res["fold"] = fold_idx
        res["fit_time"] = np.round(fit_time, 3)
        fold_metrics.append(res)

    if not fold_metrics:
        return {"error": "Cross-validation could not complete any valid folds."}

    df_results = pd.DataFrame(fold_metrics)
    summary = {
        "cv_type": cv_type,
        "n_folds_completed": len(fold_metrics),
        "mean_accuracy": float(np.round(df_results["accuracy"].mean(), 4)),
        "std_accuracy": float(np.round(df_results["accuracy"].std(), 4)),
        "mean_recall": float(np.round(df_results["recall"].mean(), 4)),
        "std_recall": float(np.round(df_results["recall"].std(), 4)),
        "mean_f1": float(np.round(df_results["f1"].mean(), 4)),
        "std_f1": float(np.round(df_results["f1"].std(), 4)),
        "folds": fold_metrics,
    }
    return summary
