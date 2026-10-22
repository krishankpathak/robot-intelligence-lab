"""
Data Validation Module for ROBOT INTELLIGENCE LAB
Ensures incoming telemetry conforms to structural, type, and physical domain constraints.
"""
from typing import Any
import pandas as pd
import numpy as np

from config.settings import (
    PRIMARY_SENSOR_FEATURES,
    TARGET_COLUMN,
    SENSOR_BOUNDS,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("data_validation")


def validate_telemetry_schema(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
    require_target: bool = False,
    target_col: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """
    Validates the structure and schema of a sensor telemetry DataFrame.
    Returns:
        dict containing 'is_valid', 'errors', 'warnings', and 'schema_info'.
    """
    if required_features is None:
        required_features = PRIMARY_SENSOR_FEATURES

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": [f"Input must be a pandas DataFrame, got {type(df)}"],
            "warnings": [],
            "schema_info": {},
        }

    if df.empty:
        return {
            "is_valid": False,
            "errors": ["DataFrame is empty (0 rows)"],
            "warnings": [],
            "schema_info": {},
        }

    # 1. Feature presence check
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        errors.append(f"Missing required sensor features: {missing_features}")

    # 2. Target check
    if require_target:
        if target_col not in df.columns:
            errors.append(f"Missing target column: '{target_col}'")
        else:
            unique_targets = df[target_col].dropna().unique()
            if not set(unique_targets).issubset({0, 1}):
                errors.append(f"Target '{target_col}' contains invalid non-binary classes: {unique_targets}")

    # 3. Numeric type validation
    present_features = [f for f in required_features if f in df.columns]
    for feat in present_features:
        if not np.issubdtype(df[feat].dtype, np.number):
            errors.append(f"Feature '{feat}' must be numeric, got type {df[feat].dtype}")

    # 4. Out-of-bounds domain physical checks
    for feat in present_features:
        if feat in SENSOR_BOUNDS and np.issubdtype(df[feat].dtype, np.number):
            bounds = SENSOR_BOUNDS[feat]
            n_below = (df[feat] < bounds["nominal_min"]).sum()
            n_above = (df[feat] > bounds.get("critical_max", bounds["nominal_max"])).sum()

            if n_below > 0:
                warnings.append(
                    f"Feature '{feat}' has {n_below} observations below nominal min ({bounds['nominal_min']})."
                )
            if n_above > 0:
                warnings.append(
                    f"Feature '{feat}' has {n_above} observations above critical limit ({bounds.get('critical_max', bounds['nominal_max'])})."
                )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    schema_info = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "has_target": target_col in df.columns,
    }

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"Validation failed with {len(errors)} errors: {errors}")

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "schema_info": schema_info,
    }


def validate_inference_input(
    input_data: dict[str, float] | pd.DataFrame,
    expected_features: list[str],
) -> tuple[bool, str, np.ndarray | None]:
    """
    Validates a single sample or batch for local inference service.
    Returns (is_valid, error_message, feature_matrix).
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        return False, f"Expected dict or DataFrame, got {type(input_data)}", None

    # Check missing features
    missing = [col for col in expected_features if col not in df_input.columns]
    if missing:
        return False, f"Missing required input features for model: {missing}", None

    # Ensure correct column order
    ordered_df = df_input[expected_features]

    # Check for NaNs
    if ordered_df.isnull().any().any():
        return False, "Input data contains NaN/null values which cannot be passed to inference", None

    try:
        features_array = ordered_df.to_numpy(dtype=np.float64)
        return True, "", features_array
    except Exception as e:
        return False, f"Failed to convert features to numeric float array: {e}", None
