"""
Sensor Simulation Engine for ROBOT INTELLIGENCE LAB
Orchestrates real-data replay and synthetic telemetry generation with bounded memory buffers
and leakage-free local model inference.
"""
from typing import Any
from collections import deque
import pandas as pd
import numpy as np

from config.settings import SIMULATION_CONFIG
from src.simulation.sensor_generator import SyntheticSensorGenerator
from src.inference.prediction_service import LocalPredictionService
from src.utils.logging_config import setup_logger

logger = setup_logger("simulation_engine")


class SensorSimulationEngine:
    """
    Simulation engine managing time progression, bounded history buffers,
    and real-time inference without label leakage.
    """

    def __init__(self, buffer_size: int = 150):
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)
        self.generator = SyntheticSensorGenerator()

        self.sim_time = 0.0
        self.step_index = 0
        self.is_paused = False
        self.mode = "SYNTHETIC"  # or 'REAL_DATA'

        # Real data replay pointer
        self.real_data_source: pd.DataFrame | None = None
        self.real_data_idx = 0

    def reset(self) -> None:
        """Restores engine to clean initial state."""
        self.history.clear()
        self.sim_time = 0.0
        self.step_index = 0
        self.real_data_idx = 0
        self.is_paused = False
        logger.info("Simulation engine reset to initial state.")

    def set_paused(self, paused: bool) -> None:
        """Pauses or resumes simulation time progression."""
        self.is_paused = paused

    def set_real_data_source(self, df: pd.DataFrame) -> None:
        """Loads a real telemetry DataFrame for authentic replay."""
        self.mode = "REAL_DATA"
        self.real_data_source = df.copy().reset_index(drop=True)
        self.real_data_idx = 0

    def set_synthetic_mode(self) -> None:
        """Switches engine to synthetic scenario generation."""
        self.mode = "SYNTHETIC"

    def step(
        self,
        scenario_key: str = "NORMAL_STEADY_STATE",
        custom_params: dict[str, float] | None = None,
        prediction_service: LocalPredictionService | None = None,
        custom_threshold: float = 0.5,
    ) -> dict[str, Any] | None:
        """
        Advances the simulation by one time step (0.1s dt) if not paused.
        Executes inference strictly on sensor features.
        """
        if self.is_paused:
            return self.history[-1] if self.history else None

        dt = SIMULATION_CONFIG["sample_dt_sec"]

        if self.mode == "REAL_DATA" and self.real_data_source is not None and not self.real_data_source.empty:
            if self.real_data_idx >= len(self.real_data_source):
                self.real_data_idx = 0  # Loop replay

            row = self.real_data_source.iloc[self.real_data_idx]
            self.real_data_idx += 1

            sample = {
                "time": float(np.round(self.sim_time, 2)),
                "position": float(row.get("position", 0.0)),
                "temperature": float(row.get("temperature", 0.0)),
                "voltage": float(row.get("voltage", 0.0)),
                "data_origin": "REAL_KAGGLE_TELEMETRY",
                "motor": str(row.get("motor", "motor_1")),
                "test_condition": str(row.get("test_condition", "unknown")),
                "actual_ground_truth_label": int(row.get("label", 0)) if "label" in row else None,
            }
        else:
            # Synthetic generation
            sample = self.generator.generate_step(
                scenario_key=scenario_key,
                sim_time=self.sim_time,
                step_index=self.step_index,
                custom_params=custom_params,
            )

        self.sim_time += dt
        self.step_index += 1

        # Model Inference: strictly passing sensor features ONLY!
        model_features = {
            "position": sample["position"],
            "temperature": sample["temperature"],
            "voltage": sample["voltage"],
        }

        if prediction_service is not None:
            pred_res = prediction_service.predict_sample(model_features, custom_threshold=custom_threshold)
            sample["model_predicted_class"] = pred_res.get("predicted_class")
            sample["model_fault_probability"] = pred_res.get("fault_probability")
            sample["model_explanation"] = pred_res.get("explanation", {})
            sample["prediction_success"] = pred_res.get("success", False)
        else:
            sample["model_predicted_class"] = None
            sample["model_fault_probability"] = None
            sample["model_explanation"] = {}
            sample["prediction_success"] = False

        self.history.append(sample)
        return sample

    def get_history_dataframe(self) -> pd.DataFrame:
        """Returns the bounded history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(list(self.history))
