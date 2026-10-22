"""
Feature Engineering Module for ROBOT INTELLIGENCE LAB
Extracts physically meaningful telemetry indicators (rates of change, thermal volatility, voltage sag)
computed strictly within motor and sequence boundaries to prevent cross-run leakage.
"""
from typing import Any
import pandas as pd
import numpy as np

from config.settings import (
    PRIMARY_SENSOR_FEATURES,
    MOTOR_COLUMN,
    SEQUENCE_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("feature_engineering")

FEATURE_DICTIONARY = {
    "position": {
        "formula": "raw sensor reading",
        "inputs": ["position"],
        "interpretation": "Current motor rotary encoder position in steps (0-999).",
        "uses_history": False,
        "leakage_risk": "None.",
    },
    "temperature": {
        "formula": "raw sensor reading",
        "inputs": ["temperature"],
        "interpretation": "Current motor coil/casing temperature in °C.",
        "uses_history": False,
        "leakage_risk": "None.",
    },
    "voltage": {
        "formula": "raw sensor reading",
        "inputs": ["voltage"],
        "interpretation": "Operational voltage across motor driver in millivolts.",
        "uses_history": False,
        "leakage_risk": "None.",
    },
    "temp_rate_of_change": {
        "formula": "(T_t - T_{t-1}) / delta_t",
        "inputs": ["temperature", "time"],
        "interpretation": "Thermal gradient / heating rate. Sharp spikes indicate mechanical jam or overcurrent.",
        "uses_history": True,
        "leakage_risk": "Calculated strictly within motor and test run sequence; zero future leakage.",
    },
    "temp_rolling_mean_5": {
        "formula": "mean(T_{t-4} ... T_t)",
        "inputs": ["temperature"],
        "interpretation": "Smoothed thermal baseline over 0.5s window.",
        "uses_history": True,
        "leakage_risk": "Causal moving window looking backward only.",
    },
    "temp_rolling_std_5": {
        "formula": "std(T_{t-4} ... T_t)",
        "inputs": ["temperature"],
        "interpretation": "Thermal volatility / thermal noise.",
        "uses_history": True,
        "leakage_risk": "Backward-only window.",
    },
    "voltage_diff_1": {
        "formula": "V_t - V_{t-1}",
        "inputs": ["voltage"],
        "interpretation": "Voltage step variation. Sudden drops reflect instantaneous load surges.",
        "uses_history": True,
        "leakage_risk": "Backward first difference.",
    },
    "voltage_rolling_std_5": {
        "formula": "std(V_{t-4} ... V_t)",
        "inputs": ["voltage"],
        "interpretation": "High-frequency electrical ripple / jitter caused by commutator wear.",
        "uses_history": True,
        "leakage_risk": "Backward-only window.",
    },
    "position_velocity": {
        "formula": "(pos_t - pos_{t-1}) / delta_t",
        "inputs": ["position", "time"],
        "interpretation": "Estimated rotary speed / movement velocity.",
        "uses_history": True,
        "leakage_risk": "Backward first difference.",
    },
}


def compute_engineered_features(
    df: pd.DataFrame,
    window_size: int = 5,
    include_rolling: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Computes time-domain engineered features within (motor, test_condition) partitions.
    Guarantees no boundary crossing between separate test runs or motors.

    Returns:
        (df_with_features, list_of_new_feature_names)
    """
    df_out = df.copy()

    # Determine grouping columns
    group_cols = []
    if MOTOR_COLUMN in df_out.columns:
        group_cols.append(MOTOR_COLUMN)
    if SEQUENCE_COLUMN in df_out.columns:
        group_cols.append(SEQUENCE_COLUMN)

    new_feature_names = []

    if not group_cols:
        # Fallback if no grouping column is present
        grouped = [("", df_out)]
    else:
        grouped = df_out.groupby(group_cols, group_keys=False)

    # 1. Temperature rate of change & velocity
    def _compute_group_features(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        dt = 0.1
        if TIMESTAMP_COLUMN in g.columns and len(g) > 1:
            diff_t = g[TIMESTAMP_COLUMN].diff().replace(0, np.nan).fillna(0.1)
            dt = diff_t

        if "temperature" in g.columns:
            g["temp_diff_1"] = g["temperature"].diff().fillna(0.0)
            g["temp_rate_of_change"] = (g["temp_diff_1"] / dt).clip(-50.0, 50.0)
            if include_rolling:
                g["temp_rolling_mean_5"] = g["temperature"].rolling(window=window_size, min_periods=1).mean()
                g["temp_rolling_std_5"] = g["temperature"].rolling(window=window_size, min_periods=1).std().fillna(0.0)

        if "voltage" in g.columns:
            g["voltage_diff_1"] = g["voltage"].diff().fillna(0.0)
            if include_rolling:
                g["voltage_rolling_std_5"] = g["voltage"].rolling(window=window_size, min_periods=1).std().fillna(0.0)

        if "position" in g.columns:
            pos_diff = g["position"].diff().fillna(0.0)
            g["position_velocity"] = (pos_diff / dt).clip(-2000.0, 2000.0)

        return g

    if group_cols:
        df_out = grouped.apply(_compute_group_features)
    else:
        df_out = _compute_group_features(df_out)

    candidate_new = [
        "temp_diff_1", "temp_rate_of_change", "temp_rolling_mean_5",
        "temp_rolling_std_5", "voltage_diff_1", "voltage_rolling_std_5",
        "position_velocity"
    ]
    new_feature_names = [f for f in candidate_new if f in df_out.columns]

    logger.info(f"Engineered {len(new_feature_names)} features: {new_feature_names}")
    return df_out, new_feature_names


def get_feature_dictionary() -> dict[str, dict[str, Any]]:
    """Returns the descriptive feature dictionary."""
    return FEATURE_DICTIONARY
