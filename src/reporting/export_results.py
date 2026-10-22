"""
Export Results Utilities for ROBOT INTELLIGENCE LAB
Exports predictions and submission files for external evaluation.
"""
from pathlib import Path
import pandas as pd


def export_predictions_to_csv(
    df_predictions: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Exports prediction results to a clean CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(p, index=False)
    return p
