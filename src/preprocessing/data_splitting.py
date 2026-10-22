"""
Data Splitting and Leakage Prevention Module for ROBOT INTELLIGENCE LAB
Provides Grouped, Temporal, and Stratified splitting strategies with explicit leakage guardrails.
"""
from typing import Any
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd
import numpy as np

from config.settings import TARGET_COLUMN, SEQUENCE_COLUMN, TIMESTAMP_COLUMN
from src.utils.logging_config import setup_logger

logger = setup_logger("data_splitting")


SPLIT_STRATEGY_EXPLANATIONS = {
    "grouped": (
        "Grouped Run Splitting: Groups samples by operational run ('test_condition'). "
        "Entire test runs are held out to evaluate how well the model generalizes to completely "
        "unseen operational sessions. This strictly prevents temporal data leakage between contiguous time samples."
    ),
    "temporal": (
        "Temporal Chronological Splitting: Sorts sequences chronologically. The earlier runs form the "
        "training set and the most recent runs form the test set. Emulates real-world deployment where "
        "historical data trains models to forecast future robot health."
    ),
    "stratified": (
        "Random Stratified Splitting: Randomly samples observations preserving overall class balance. "
        "WARNING: In time-series robotics telemetry, randomly shuffling adjacent time-steps (10 Hz) "
        "causes significant temporal leakage between training and testing samples."
    ),
}


def split_telemetry_data(
    df: pd.DataFrame,
    strategy: str = "grouped",
    test_size: float = 0.25,
    random_state: int = 42,
    target_col: str = TARGET_COLUMN,
    group_col: str = SEQUENCE_COLUMN,
) -> dict[str, Any]:
    """
    Partitions the dataset into training and testing subsets using the chosen strategy.

    Returns:
        dict containing:
            'train_df': pd.DataFrame,
            'test_df': pd.DataFrame,
            'strategy': str,
            'explanation': str,
            'train_runs': list[str],
            'test_runs': list[str],
            'train_classes': dict,
            'test_classes': dict,
            'leakage_warning': str | None
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_runs: list[str] = []
    test_runs: list[str] = []
    leakage_warning: str | None = None

    if strategy == "grouped":
        if group_col not in df.columns:
            logger.warning(f"Group column '{group_col}' missing; falling back to stratified split.")
            strategy = "stratified"
        else:
            groups = df[group_col].astype(str)
            unique_groups = groups.unique()

            # Ensure runs with faults are distributed across both train and test if possible
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(df, groups=groups))

            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            test_df = df.iloc[test_idx].copy().reset_index(drop=True)

            train_runs = list(train_df[group_col].unique())
            test_runs = list(test_df[group_col].unique())

            # Check if test set contains positive faults
            if test_df[target_col].sum() == 0:
                logger.warning("Grouped split produced test partition with zero positive fault samples.")

    elif strategy == "temporal":
        if group_col in df.columns:
            ordered_runs = sorted(df[group_col].unique().tolist())
            n_test_runs = max(1, int(len(ordered_runs) * test_size))
            split_point = len(ordered_runs) - n_test_runs

            train_run_ids = ordered_runs[:split_point]
            test_run_ids = ordered_runs[split_point:]

            train_df = df[df[group_col].isin(train_run_ids)].copy().reset_index(drop=True)
            test_df = df[df[group_col].isin(test_run_ids)].copy().reset_index(drop=True)

            train_runs = train_run_ids
            test_runs = test_run_ids
        else:
            # Sort by time
            df_sorted = df.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)
            n_train = int(len(df_sorted) * (1.0 - test_size))
            train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
            test_df = df_sorted.iloc[n_train:].copy().reset_index(drop=True)

    elif strategy == "stratified":
        leakage_warning = (
            "Caution: Stratified random split was selected. Samples were randomly shuffled. "
            "Because telemetry is sampled at 10 Hz, adjacent points in the same physical fault episode "
            "may appear in both train and test partitions, causing significant temporal leakage and inflating reported performance."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        if group_col in df.columns:
            train_runs = list(train_df[group_col].unique())
            test_runs = list(test_df[group_col].unique())
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'. Supported: 'grouped', 'temporal', 'stratified'")

    train_classes = train_df[target_col].value_counts().to_dict()
    test_classes = test_df[target_col].value_counts().to_dict()

    logger.info(
        f"Split complete [{strategy}]: Train rows={len(train_df)} ({train_classes}), "
        f"Test rows={len(test_df)} ({test_classes})"
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "strategy": strategy,
        "explanation": SPLIT_STRATEGY_EXPLANATIONS.get(strategy, ""),
        "train_runs": train_runs,
        "test_runs": test_runs,
        "train_classes": train_classes,
        "test_classes": test_classes,
        "leakage_warning": leakage_warning,
    }
