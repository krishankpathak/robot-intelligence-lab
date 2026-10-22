"""
Statistical Analysis Module for ROBOT INTELLIGENCE LAB
Computes univariate and bivariate statistics, class-stratified metrics, and correlation matrices.
"""
from typing import Any
import pandas as pd
import numpy as np

from config.settings import PRIMARY_SENSOR_FEATURES, TARGET_COLUMN


def compute_univariate_summary(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """Computes mean, std, median, IQR, skewness, min, max per feature."""
    rows = []
    for col in numeric_features:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            s = df[col].dropna()
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            rows.append({
                "feature": col,
                "mean": float(np.round(s.mean(), 2)),
                "std": float(np.round(s.std(), 2)),
                "median": float(np.round(s.median(), 2)),
                "iqr": float(np.round(q75 - q25, 2)),
                "skewness": float(np.round(s.skew(), 2)),
                "min": float(np.round(s.min(), 2)),
                "max": float(np.round(s.max(), 2)),
            })
    return pd.DataFrame(rows)


def compute_class_stratified_stats(
    df: pd.DataFrame,
    numeric_features: list[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Computes mean and std separated by Normal (0) vs Fault (1) classes."""
    if target_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(target_col)[numeric_features].agg(["mean", "std"])
    flat_rows = []
    for feat in numeric_features:
        mean_0 = grouped.loc[0, (feat, "mean")] if 0 in grouped.index else np.nan
        std_0 = grouped.loc[0, (feat, "std")] if 0 in grouped.index else np.nan
        mean_1 = grouped.loc[1, (feat, "mean")] if 1 in grouped.index else np.nan
        std_1 = grouped.loc[1, (feat, "std")] if 1 in grouped.index else np.nan

        # Difference in means and Cohen's d effect size
        diff = mean_1 - mean_0
        pooled_std = np.sqrt((std_0**2 + std_1**2) / 2.0) if not np.isnan(std_0) and not np.isnan(std_1) else np.nan
        cohen_d = diff / pooled_std if pooled_std and pooled_std > 0 else np.nan

        flat_rows.append({
            "feature": feat,
            "normal_mean (Class 0)": np.round(mean_0, 2),
            "normal_std": np.round(std_0, 2),
            "fault_mean (Class 1)": np.round(mean_1, 2),
            "fault_std": np.round(std_1, 2),
            "mean_delta (Class 1 - 0)": np.round(diff, 2),
            "effect_size_cohens_d": np.round(cohen_d, 2),
        })
    return pd.DataFrame(flat_rows)


def compute_correlation_matrix(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """Computes Pearson correlation matrix between sensor features and target."""
    cols = [c for c in numeric_features if c in df.columns]
    return df[cols].corr().round(3)
