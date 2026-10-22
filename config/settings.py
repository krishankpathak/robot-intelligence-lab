"""
Configuration and Settings for ROBOT INTELLIGENCE LAB
Local AI-Powered Predictive Maintenance and Fault Analytics
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Raw Kaggle Dataset Path
RAW_KAGGLE_DIR = Path("D:/ai_analysis/Data/robot-predictive-maintenance")
RAW_TRAIN_DIR = RAW_KAGGLE_DIR / "training_data" / "training_data"
RAW_TEST_DIR = RAW_KAGGLE_DIR / "testing_data" / "testing_data"

# Reproducibility
RANDOM_SEED = 42

# Target & Feature Definitions
TARGET_COLUMN = "label"
TIMESTAMP_COLUMN = "time"
MOTOR_COLUMN = "motor"
SEQUENCE_COLUMN = "test_condition"

PRIMARY_SENSOR_FEATURES = ["position", "temperature", "voltage"]

ALL_MOTORS = [f"motor_{i}" for i in range(1, 7)]

# Multi-motor wide feature names
WIDE_SENSOR_FEATURES = [
    f"data_motor_{m}_{feat}"
    for m in range(1, 7)
    for feat in ["position", "temperature", "voltage"]
]
WIDE_LABEL_COLUMNS = [f"data_motor_{m}_label" for m in range(1, 7)]

# Sensor Domain Ranges (Nominal & Physical Bounds)
SENSOR_BOUNDS = {
    "position": {
        "nominal_min": 0.0,
        "nominal_max": 999.0,
        "unit": "steps",
        "description": "Rotary position / encoder steps (0 - 999)"
    },
    "temperature": {
        "nominal_min": 20.0,
        "nominal_max": 65.0,
        "critical_max": 100.0,
        "unit": "°C",
        "description": "Internal coil and housing temperature"
    },
    "voltage": {
        "nominal_min": 6800.0,
        "nominal_max": 7600.0,
        "critical_min": 6000.0,
        "unit": "mV",
        "description": "Operating voltage in millivolts (nominal ~7.2V)"
    }
}

# Simulation settings
SIMULATION_CONFIG = {
    "buffer_size": 150,
    "default_speed_hz": 5,
    "sample_dt_sec": 0.1,
}

# Logging configuration
LOG_LEVEL = "INFO"
