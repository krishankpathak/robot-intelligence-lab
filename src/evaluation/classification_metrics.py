"""
Comprehensive Classification Metrics and Evaluation Module for ROBOT INTELLIGENCE LAB
Focuses on honest evaluation under extreme class imbalance (PR-AUC, ROC-AUC, Recall/F1 at custom thresholds).
"""
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.utils.logging_config import setup_logger

logger = setup_logger("classification_metrics")


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Computes a comprehensive set of classification metrics for binary fault detection.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    pr_auc = None

    if y_prob is not None:
        p1 = np.asarray(y_prob)[:, 1] if y_prob.ndim == 2 else np.asarray(y_prob)
        # Check if both classes are present
        if len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, p1))
                pr_auc = float(average_precision_score(y_t, p1))
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR AUC: {e}")

    return {
        "accuracy": np.round(acc, 4),
        "precision": np.round(prec, 4),
        "recall": np.round(rec, 4),
        "f1": np.round(f1, 4),
        "balanced_accuracy": np.round(bal_acc, 4),
        "specificity": np.round(specificity, 4),
        "roc_auc": np.round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": np.round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "total_samples": len(y_t),
        "actual_faults": int(np.sum(y_t == 1)),
        "predicted_faults": int(np.sum(y_p == 1)),
    }


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluates precision, recall, and F1 across multiple decision thresholds.
    Allows operational operators to choose their risk-tolerance trade-off.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    p1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    rows = []
    for th in thresholds:
        y_pred_th = (p1 >= th).astype(int)
        res = evaluate_classifier_predictions(y_true, y_pred_th, p1)
        rows.append({
            "threshold": th,
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "false_negatives": res["confusion_matrix"]["fn"],
            "false_positives": res["confusion_matrix"]["fp"],
        })
    return pd.DataFrame(rows)


def compute_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Generates coordinate arrays for ROC and PR curves."""
    p1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, roc_th = roc_curve(y_true, p1)
    prec, rec, pr_th = precision_recall_curve(y_true, p1)

    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_th.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pr_th.tolist()},
    }


def build_comparison_leaderboard(results_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Formats multiple evaluation results into a sorted leaderboard DataFrame."""
    rows = []
    for model_name, metrics in results_dict.items():
        cm = metrics.get("confusion_matrix", {})
        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy"),
            "Precision": metrics.get("precision"),
            "Recall (Fault Detection)": metrics.get("recall"),
            "F1-Score": metrics.get("f1"),
            "Balanced Acc": metrics.get("balanced_accuracy"),
            "PR-AUC": metrics.get("pr_auc"),
            "ROC-AUC": metrics.get("roc_auc"),
            "False Negatives (Missed)": cm.get("fn"),
            "False Positives": cm.get("fp"),
            "Fit Time (s)": metrics.get("fit_time_sec"),
        })
    df_board = pd.DataFrame(rows)
    if "Recall (Fault Detection)" in df_board.columns:
        df_board = df_board.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    return df_board
