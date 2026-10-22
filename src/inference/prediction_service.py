"""
Local Model Prediction Service for ROBOT INTELLIGENCE LAB
Provides 100% offline, local model inference with strict schema validation and deterministic explanations.
"""
from typing import Any
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.model_registry import load_model_artifact
from src.data.data_validation import validate_inference_input
from src.explainability.local_explanation import explain_single_prediction
from src.utils.logging_config import setup_logger

logger = setup_logger("prediction_service")


class LocalPredictionService:
    """
    Production-grade local prediction service.
    Loads a verified model artifact and serves real-time or batch predictions.
    """

    def __init__(
        self,
        model_artifact_path: Path | str,
        training_baseline_stats: dict[str, dict[str, float]] | None = None,
        decision_threshold: float = 0.5,
    ):
        self.pipeline, self.metadata = load_model_artifact(model_artifact_path)
        self.expected_features: list[str] = self.metadata.get("feature_names", ["position", "temperature", "voltage"])
        self.model_name: str = self.metadata.get("model_name", "Unknown Model")
        self.training_stats = training_baseline_stats or {}
        self.decision_threshold = decision_threshold

    def predict_sample(
        self,
        sample: dict[str, float],
        custom_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Infers fault classification for a single telemetry reading with schema validation and explanation.
        """
        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold

        is_valid, err_msg, feature_matrix = validate_inference_input(sample, self.expected_features)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "predicted_class": None,
                "fault_probability": None,
            }

        df_input = pd.DataFrame([sample])[self.expected_features]

        # Predict probability if supported
        prob_fault = None
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(df_input)
                prob_fault = float(probs[0, 1])
                predicted_class = int(prob_fault >= threshold)
            except Exception:
                predicted_class = int(self.pipeline.predict(df_input)[0])
        else:
            predicted_class = int(self.pipeline.predict(df_input)[0])

        # Local explanation
        explanation = explain_single_prediction(
            sample_features=sample,
            predicted_class=predicted_class,
            predicted_prob=prob_fault,
            training_baseline_stats=self.training_stats,
            threshold=threshold,
        )

        return {
            "success": True,
            "predicted_class": predicted_class,
            "is_fault": predicted_class == 1,
            "fault_probability": prob_fault,
            "threshold_used": threshold,
            "explanation": explanation,
        }

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        custom_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Runs batch inference over a DataFrame."""
        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold

        missing = [f for f in self.expected_features if f not in df_batch.columns]
        if missing:
            raise ValueError(f"Batch DataFrame missing expected features: {missing}")

        X = df_batch[self.expected_features]
        out_df = df_batch.copy()

        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(X)
                out_df["fault_probability"] = probs[:, 1]
                out_df["predicted_label"] = (probs[:, 1] >= threshold).astype(int)
                return out_df
            except Exception:
                pass

        out_df["predicted_label"] = self.pipeline.predict(X).astype(int)
        out_df["fault_probability"] = np.nan
        return out_df
