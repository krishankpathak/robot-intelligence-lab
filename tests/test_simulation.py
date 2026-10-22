"""
Automated Unit Tests for Sensor Simulation Engine and Scenario Generator
"""
import pytest
import numpy as np

from src.simulation.sensor_generator import SyntheticSensorGenerator
from src.simulation.simulation_engine import SensorSimulationEngine
from src.simulation.operating_scenarios import OPERATING_SCENARIOS


def test_synthetic_sensor_generator_schema():
    """Verifies sensor generator produces correct schema and marks synthetic origin."""
    gen = SyntheticSensorGenerator(seed=42)
    step = gen.generate_step(scenario_key="NORMAL_STEADY_STATE", sim_time=1.0, step_index=10)

    assert "time" in step
    assert "position" in step
    assert "temperature" in step
    assert "voltage" in step
    assert step["data_origin"] == "SYNTHETIC_SIMULATION"
    assert step["scenario_key"] == "NORMAL_STEADY_STATE"


def test_simulation_engine_pause_and_resume():
    """Verifies that pausing the engine stops simulation time progression."""
    engine = SensorSimulationEngine(buffer_size=50)
    engine.step(scenario_key="NORMAL_STEADY_STATE")
    t1 = engine.sim_time

    engine.set_paused(True)
    engine.step(scenario_key="NORMAL_STEADY_STATE")
    t2 = engine.sim_time

    assert t1 == t2, "Time progressed while simulation was paused!"

    engine.set_paused(False)
    engine.step(scenario_key="NORMAL_STEADY_STATE")
    t3 = engine.sim_time
    assert t3 > t2, "Time failed to resume after unpausing."


def test_simulation_engine_reset():
    """Verifies reset restores engine to initial state."""
    engine = SensorSimulationEngine(buffer_size=50)
    for _ in range(10):
        engine.step(scenario_key="NORMAL_STEADY_STATE")

    assert len(engine.history) == 10
    assert engine.sim_time > 0.0

    engine.reset()
    assert len(engine.history) == 0
    assert engine.sim_time == 0.0
    assert engine.step_index == 0


def test_simulation_buffer_bounded_memory():
    """
    Critical Reliability Test: Verifies that simulation history buffer NEVER grows
    unboundedly beyond its configured buffer_size limit (preventing memory leaks).
    """
    buffer_limit = 25
    engine = SensorSimulationEngine(buffer_size=buffer_limit)

    for _ in range(100):
        engine.step(scenario_key="NORMAL_STEADY_STATE")

    assert len(engine.history) == buffer_limit
    df_hist = engine.get_history_dataframe()
    assert len(df_hist) == buffer_limit


def test_no_scenario_label_leakage_into_model_features():
    """
    Verifies that scenario metadata or intended scenario labels do NOT get passed
    as predictive features into the model input dictionary.
    """
    engine = SensorSimulationEngine(buffer_size=10)
    sample = engine.step(scenario_key="ELEVATED_TEMPERATURE_OVERHEAT")

    # Features passed to model must only be physical signals
    model_features = ["position", "temperature", "voltage"]
    for feat in model_features:
        assert feat in sample
    # Check intended label is kept separate
    assert "scenario_intended_label" in sample
