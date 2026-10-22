"""
Synthetic Sensor Generator for ROBOT INTELLIGENCE LAB
Produces realistic, continuous telemetry streams based on scenario configurations.
All outputs explicitly labeled data_origin='SYNTHETIC_SIMULATION'.
"""
from typing import Any
import numpy as np

from src.simulation.operating_scenarios import OPERATING_SCENARIOS
from src.simulation.fault_injection import inject_thermal_fault, inject_voltage_sag, inject_sensor_glitch


class SyntheticSensorGenerator:
    """Generates continuous synthetic robot telemetry for simulation."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_step(
        self,
        scenario_key: str,
        sim_time: float,
        step_index: int,
        custom_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generates a single time-step observation."""
        if scenario_key == "CUSTOM_USER_SCENARIO" and custom_params:
            pos = custom_params.get("position", 500.0) + self.rng.normal(0, 1.0)
            temp = custom_params.get("temperature", 35.0) + self.rng.normal(0, 0.2)
            volt = custom_params.get("voltage", 7200.0) + self.rng.normal(0, 5.0)
            intended_label = custom_params.get("intended_label", -1)
        else:
            cfg = OPERATING_SCENARIOS.get(scenario_key, OPERATING_SCENARIOS["NORMAL_STEADY_STATE"])
            intended_label = cfg.get("expected_label", 0)

            # Position
            omega = 0.5  # rad/s
            pos = cfg["position_offset"] + cfg["position_amplitude"] * np.sin(omega * sim_time) + self.rng.normal(0, 2.0)

            # Temperature
            temp = cfg["temp_base"] + (step_index * cfg.get("temp_drift", 0.0) * 0.05) + self.rng.normal(0, 0.3)

            # Voltage
            volt = cfg["voltage_base"] + self.rng.normal(0, cfg["voltage_noise_std"])

            if cfg.get("inject_glitches", False):
                if self.rng.random() < 0.1:
                    pos = self.rng.choice([-32000.0, 999.0])
                if self.rng.random() < 0.1:
                    volt = -25000.0

        return {
            "time": float(np.round(sim_time, 2)),
            "position": float(np.round(pos, 1)),
            "temperature": float(np.round(temp, 1)),
            "voltage": float(np.round(volt, 1)),
            "data_origin": "SYNTHETIC_SIMULATION",
            "scenario_key": scenario_key,
            "scenario_intended_label": intended_label,
        }
