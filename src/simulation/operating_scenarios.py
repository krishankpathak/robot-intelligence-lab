"""
Operating Scenarios Catalog for Sensor Simulation in ROBOT INTELLIGENCE LAB
Defines synthetic operational modes and fault injection rules with explicit synthetic labeling.
"""
from typing import Any

OPERATING_SCENARIOS: dict[str, dict[str, Any]] = {
    "NORMAL_STEADY_STATE": {
        "display_name": "1. Normal Steady-State Operation",
        "description": "Nominal sinusoidal position cycling, baseline temperature (30-35°C), stable voltage (~7.2V).",
        "expected_label": 0,
        "is_fault_intended": False,
        "temp_base": 32.0,
        "temp_drift": 0.0,
        "voltage_base": 7220.0,
        "voltage_noise_std": 20.0,
        "position_amplitude": 400.0,
        "position_offset": 500.0,
    },
    "ELEVATED_TEMPERATURE_OVERHEAT": {
        "display_name": "2. High Friction & Overheating Fault",
        "description": "Progressive thermal rise (reaching 48-65°C) caused by bearing friction or lack of lubrication.",
        "expected_label": 1,
        "is_fault_intended": True,
        "temp_base": 42.0,
        "temp_drift": 1.2,
        "voltage_base": 7120.0,
        "voltage_noise_std": 30.0,
        "position_amplitude": 350.0,
        "position_offset": 520.0,
    },
    "VOLTAGE_SAG_HEAVY_LOAD": {
        "display_name": "3. Electrical Degradation & Voltage Sag",
        "description": "Operating voltage sags by ~300mV with expanded ripple variance due to winding deterioration.",
        "expected_label": 1,
        "is_fault_intended": True,
        "temp_base": 39.0,
        "temp_drift": 0.2,
        "voltage_base": 6880.0,
        "voltage_noise_std": 90.0,
        "position_amplitude": 300.0,
        "position_offset": 500.0,
    },
    "MECHANICAL_STALL": {
        "display_name": "4. Mechanical Jam / Arm Stalling",
        "description": "Motor locked at high displacement; rapid Joule heating with sharp current surge.",
        "expected_label": 1,
        "is_fault_intended": True,
        "temp_base": 45.0,
        "temp_drift": 1.5,
        "voltage_base": 6750.0,
        "voltage_noise_std": 60.0,
        "position_amplitude": 0.0,
        "position_offset": 880.0,
    },
    "SENSOR_GLITCH_SPIKE": {
        "display_name": "5. Telemetry Noise & Dropout Glitch",
        "description": "Transient negative voltage and position dropout spikes caused by electrical EMI on serial bus.",
        "expected_label": 1,
        "is_fault_intended": True,
        "temp_base": 34.0,
        "temp_drift": 0.0,
        "voltage_base": 7200.0,
        "voltage_noise_std": 40.0,
        "position_amplitude": 400.0,
        "position_offset": 500.0,
        "inject_glitches": True,
    },
    "CUSTOM_USER_SCENARIO": {
        "display_name": "6. Custom User-Configured Telemetry",
        "description": "User dynamically selects position, temperature, and voltage sliders to probe model response.",
        "expected_label": -1,
        "is_fault_intended": None,
    },
}


def get_scenario_catalog() -> dict[str, dict[str, Any]]:
    """Returns the full scenario catalog."""
    return OPERATING_SCENARIOS
