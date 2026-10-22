"""
Educational Confusion Matrix and Metrics Implemented from Scratch using NumPy
Provides transparent calculation of TP, FP, TN, FN, Accuracy, Precision, Recall, and F1 Score
along with automated validation against scikit-learn.
"""
from typing import Any
import numpy as np
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score as sk_accuracy_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    f1_score as sk_f1_score,
)


def compute_scratch_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """
    Computes confusion matrix entries manually using boolean vectorized indexing.
    Returns:
        {'tp': int, 'fp': int, 'tn': int, 'fn': int}
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Lengths differ: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_scratch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculates classification metrics from raw counts with zero-division handling.
    """
    cm = compute_scratch_confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(np.round(accuracy, 4)),
        "precision": float(np.round(precision, 4)),
        "recall": float(np.round(recall, 4)),
        "f1": float(np.round(f1, 4)),
    }


def verify_against_sklearn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """
    Compares the from-scratch metrics with scikit-learn implementations on the exact same predictions.
    Returns:
        dict with comparison values, delta, and pass/fail boolean.
    """
    custom = compute_scratch_metrics(y_true, y_pred)

    sk_tn, sk_fp, sk_fn, sk_tp = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sk_acc = float(sk_accuracy_score(y_true, y_pred))
    sk_prec = float(sk_precision_score(y_true, y_pred, zero_division=0))
    sk_rec = float(sk_recall_score(y_true, y_pred, zero_division=0))
    sk_f1_val = float(sk_f1_score(y_true, y_pred, zero_division=0))

    checks = {
        "tp_match": custom["tp"] == int(sk_tp),
        "fp_match": custom["fp"] == int(sk_fp),
        "tn_match": custom["tn"] == int(sk_tn),
        "fn_match": custom["fn"] == int(sk_fn),
        "accuracy_diff": abs(custom["accuracy"] - sk_acc),
        "precision_diff": abs(custom["precision"] - sk_prec),
        "recall_diff": abs(custom["recall"] - sk_rec),
        "f1_diff": abs(custom["f1"] - sk_f1_val),
    }

    all_match = (
        checks["tp_match"]
        and checks["fp_match"]
        and checks["tn_match"]
        and checks["fn_match"]
        and checks["accuracy_diff"] <= tolerance
        and checks["precision_diff"] <= tolerance
        and checks["recall_diff"] <= tolerance
        and checks["f1_diff"] <= tolerance
    )

    return {
        "all_match": all_match,
        "custom_metrics": custom,
        "sklearn_metrics": {
            "tp": int(sk_tp),
            "fp": int(sk_fp),
            "tn": int(sk_tn),
            "fn": int(sk_fn),
            "accuracy": np.round(sk_acc, 4),
            "precision": np.round(sk_prec, 4),
            "recall": np.round(sk_rec, 4),
            "f1": np.round(sk_f1_val, 4),
        },
        "checks": checks,
    }
