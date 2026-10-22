"""
Data Quality Audit Module for ROBOT INTELLIGENCE LAB
Provides automated data health checks, class imbalance analysis, and anomaly scanning.
"""
from typing import Any
import pandas as pd
import numpy as np

from config.settings import TARGET_COLUMN


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Computes missing value counts and percentages per column."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    report = pd.DataFrame({
        "missing_count": null_counts,
        "missing_percentage": np.round(null_pct, 3),
        "data_type": df.dtypes.astype(str),
    })
    return report.sort_values(by="missing_count", ascending=False)


def analyze_class_imbalance(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict[str, Any]:
    """Analyzes class distribution and imbalance ratio for the target column."""
    if target_col not in df.columns:
        return {"has_target": False, "error": f"Target column '{target_col}' not found"}

    counts = df[target_col].value_counts().to_dict()
    total = len(df[target_col].dropna())
    if total == 0:
        return {"has_target": True, "error": "Target column has no non-null entries"}

    percentages = {k: np.round((v / total) * 100, 3) for k, v in counts.items()}

    sorted_classes = sorted(counts.items(), key=lambda x: x[1])
    minority_class, minority_count = sorted_classes[0]
    majority_class, majority_count = sorted_classes[-1]

    imbalance_ratio = np.round(majority_count / max(minority_count, 1), 2)

    return {
        "has_target": True,
        "counts": counts,
        "percentages": percentages,
        "total_valid_samples": total,
        "minority_class": minority_class,
        "minority_count": minority_count,
        "majority_class": majority_class,
        "majority_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 3.0,
    }


def detect_constant_features(df: pd.DataFrame, threshold: float = 0.0) -> list[str]:
    """Detects constant or near-zero variance numeric features."""
    constant_features = []
    num_df = df.select_dtypes(include=[np.number])
    for col in num_df.columns:
        if num_df[col].nunique() <= 1:
            constant_features.append(col)
        elif num_df[col].std() <= threshold:
            constant_features.append(col)
    return constant_features


def detect_outliers_iqr(df: pd.DataFrame, numeric_features: list[str]) -> dict[str, int]:
    """Identifies outlier counts per numeric feature using the standard 1.5 * IQR rule."""
    outlier_counts = {}
    for col in numeric_features:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            outlier_counts[col] = n_outliers
    return outlier_counts


def run_full_quality_audit(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict[str, Any]:
    """Runs a complete automated data quality audit and returns structured metrics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_report = analyze_missing_values(df)
    imbalance_report = analyze_class_imbalance(df, target_col)
    constant_cols = detect_constant_features(df)
    duplicate_rows = int(df.duplicated().sum())
    outlier_counts = detect_outliers_iqr(df, [c for c in numeric_cols if c != target_col])

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": duplicate_rows,
        "has_missing_values": int(missing_report["missing_count"].sum()) > 0,
        "missing_report": missing_report.to_dict(orient="index"),
        "constant_columns": constant_cols,
        "imbalance_report": imbalance_report,
        "outlier_counts": outlier_counts,
    }
