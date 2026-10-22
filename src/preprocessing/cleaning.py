"""
Data Cleaning and Sanitization Module for ROBOT INTELLIGENCE LAB
Handles missing values, duplicate checks, and sensor glitch sanitation.
"""
import pandas as pd
import numpy as np

from config.settings import PRIMARY_SENSOR_FEATURES, SENSOR_BOUNDS
from src.utils.logging_config import setup_logger

logger = setup_logger("cleaning")


def clean_telemetry_data(
    df: pd.DataFrame,
    handle_glitches: bool = True,
    impute_strategy: str = "median",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Cleans raw sensor telemetry:
    1. Removes exact duplicates.
    2. Treats extreme physical glitches (e.g. negative position or voltage dropouts).
    3. Imputes missing values if any exist.

    Returns:
        (cleaned_df, report_dict)
    """
    cleaned = df.copy()
    report = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "glitches_clipped": 0,
        "nans_imputed": 0,
    }

    # 1. Deduplication
    init_len = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    report["duplicates_removed"] = init_len - len(cleaned)

    # 2. Extreme sensor glitch sanitation
    # E.g. negative encoder readings (-32271) caused by serial bus corruption
    if handle_glitches:
        for feat in PRIMARY_SENSOR_FEATURES:
            if feat in cleaned.columns and np.issubdtype(cleaned[feat].dtype, np.number):
                bounds = SENSOR_BOUNDS.get(feat, {})
                min_valid = bounds.get("nominal_min", 0.0)
                # Count extreme drops below physical limit
                extreme_low_mask = cleaned[feat] < (min_valid - 50.0)
                glitch_count = int(extreme_low_mask.sum())
                if glitch_count > 0:
                    report["glitches_clipped"] += glitch_count
                    # Replace extreme negative drop with median of non-glitch data
                    valid_median = cleaned.loc[~extreme_low_mask, feat].median()
                    cleaned.loc[extreme_low_mask, feat] = valid_median
                    logger.info(f"Replaced {glitch_count} extreme glitch dropouts in '{feat}' with median ({valid_median:.2f}).")

    # 3. Missing value imputation
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    null_counts = cleaned[num_cols].isnull().sum().sum()
    if null_counts > 0:
        report["nans_imputed"] = int(null_counts)
        for col in num_cols:
            if cleaned[col].isnull().any():
                val = cleaned[col].median() if impute_strategy == "median" else cleaned[col].mean()
                cleaned[col] = cleaned[col].fillna(val)

    report["final_rows"] = len(cleaned)
    return cleaned, report
