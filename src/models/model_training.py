"""
Model Training and Orchestration Module for ROBOT INTELLIGENCE LAB
Trains candidate algorithms through unified pipelines, preventing leakage and capturing honest metrics.
"""
from typing import Any
import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from config.settings import RANDOM_SEED, PRIMARY_SENSOR_FEATURES, TARGET_COLUMN
from src.preprocessing.transformations import build_preprocessor
from src.models.baseline import create_baseline_model
from src.models.logistic_regression_model import create_logistic_regression_model
from src.models.decision_tree_model import create_decision_tree_model
from src.models.random_forest_model import create_random_forest_model
from src.models.gradient_boosting_model import create_gradient_boosting_model
from src.evaluation.classification_metrics import evaluate_classifier_predictions
from src.utils.logging_config import setup_logger

logger = setup_logger("model_training")


SUPPORTED_MODEL_TYPES = [
    "Dummy Classifier (Baseline)",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    scaler_type: str = "standard",
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Constructs a Pipeline with preprocessor and estimator, fits on X_train,
    and evaluates on X_test.
    """
    preprocessor = build_preprocessor(numeric_features=feature_names, scaler_type=scaler_type)

    if model_name == "Dummy Classifier (Baseline)":
        clf = create_baseline_model(strategy="most_frequent", random_state=random_state)
    elif model_name == "Logistic Regression":
        clf = create_logistic_regression_model(C=1.0, class_weight="balanced", random_state=random_state)
    elif model_name == "Decision Tree":
        clf = create_decision_tree_model(max_depth=6, class_weight="balanced", random_state=random_state)
    elif model_name == "Random Forest":
        clf = create_random_forest_model(n_estimators=100, max_depth=8, class_weight="balanced_subsample", random_state=random_state)
    elif model_name == "Gradient Boosting":
        clf = create_gradient_boosting_model(model_variant="hist", class_weight="balanced", random_state=random_state)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from {SUPPORTED_MODEL_TYPES}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    # Measure fit time
    t0 = time.perf_counter()
    pipeline.fit(X_train[feature_names], y_train)
    fit_time = time.perf_counter() - t0

    # Measure inference latency on test set
    t_inf_start = time.perf_counter()
    y_pred = pipeline.predict(X_test[feature_names])
    inf_time = (time.perf_counter() - t_inf_start) / max(len(X_test), 1)

    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test[feature_names])
        except Exception as e:
            logger.warning(f"Failed to compute predict_proba for {model_name}: {e}")

    metrics = evaluate_classifier_predictions(y_test, y_pred, y_prob)
    metrics["fit_time_sec"] = float(np.round(fit_time, 4))
    metrics["inference_latency_ms"] = float(np.round(inf_time * 1000.0, 4))
    metrics["model_name"] = model_name

    logger.info(
        f"Trained {model_name}: FitTime={fit_time:.2f}s, Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, PR-AUC={metrics.get('pr_auc')}"
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_names": feature_names,
        "model_name": model_name,
    }


def train_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
    scaler_type: str = "standard",
    random_state: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """
    Trains all standard models on the supplied train partition and evaluates on test partition.
    """
    if feature_names is None:
        feature_names = PRIMARY_SENSOR_FEATURES

    X_train = train_df[feature_names]
    y_train = train_df[target_col].to_numpy()
    X_test = test_df[feature_names]
    y_test = test_df[target_col].to_numpy()

    all_results = {}
    for model_name in SUPPORTED_MODEL_TYPES:
        res = train_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            scaler_type=scaler_type,
            random_state=random_state,
        )
        all_results[model_name] = res

    return all_results
