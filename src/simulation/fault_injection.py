"""
Synthetic Fault Injection Rules for ROBOT INTELLIGENCE LAB
Modulates physical sensor signals to emulate overheating, electrical sag, and sensor dropout anomalies.
"""
import numpy as np


def inject_thermal_fault(current_temp: float, step: int, drift_rate: float = 0.5) -> float:
    """Simulates thermal buildup over time due to friction or continuous load."""
    return current_temp + (step * drift_rate) + float(np.random.normal(0, 0.4))


def inject_voltage_sag(base_voltage: float, sag_mv: float = 300.0, ripple_std: float = 80.0) -> float:
    """Simulates voltage drop across motor coils and driver circuitry."""
    return base_voltage - sag_mv + float(np.random.normal(0, ripple_std))


def inject_sensor_glitch(reading: float, glitch_value: float = -32000.0, probability: float = 0.15) -> float:
    """Randomly injects transient bus corruption glitches."""
    if np.random.random() < probability:
        return glitch_value
    return reading
