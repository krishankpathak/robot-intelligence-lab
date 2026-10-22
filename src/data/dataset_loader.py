"""
Dataset Loader for Robot Predictive Maintenance
Loads real multi-motor robot sensor sequences and provides documented synthetic sample generation.
"""
from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np

from config.settings import (
    RAW_TRAIN_DIR,
    RAW_TEST_DIR,
    SAMPLE_DATA_DIR,
    ALL_MOTORS,
    PRIMARY_SENSOR_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    MOTOR_COLUMN,
    SEQUENCE_COLUMN,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("dataset_loader")


def load_test_conditions(train: bool = True) -> pd.DataFrame:
    """Loads the test conditions Excel file for train or test runs."""
    base_dir = RAW_TRAIN_DIR if train else RAW_TEST_DIR
    excel_path = base_dir / "Test conditions.xlsx"
    if not excel_path.exists():
        logger.warning(f"Test conditions file not found at {excel_path}")
        return pd.DataFrame()
    return pd.read_excel(excel_path)


def load_motor_level_dataset(
    train_dir: Path | str = RAW_TRAIN_DIR,
    max_runs: int | None = None,
    use_sample_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Loads all run folders into a normalized motor-level DataFrame:
    Columns: [time, position, temperature, voltage, label, motor, test_condition]
    """
    train_path = Path(train_dir)
    if not train_path.exists() or not any(train_path.iterdir()):
        if use_sample_if_missing:
            logger.info("Raw Kaggle dataset path not found. Falling back to documented sample dataset.")
            return load_sample_dataset()
        raise FileNotFoundError(f"Training directory does not exist: {train_path}")

    run_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    if max_runs:
        run_dirs = run_dirs[:max_runs]

    all_dfs = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        csv_files = sorted(glob.glob(str(run_dir / "data_motor_*.csv")))
        for csv_path in csv_files:
            motor_name = Path(csv_path).stem.replace("data_", "")
            try:
                df_motor = pd.read_csv(csv_path)
                df_motor[MOTOR_COLUMN] = motor_name
                df_motor[SEQUENCE_COLUMN] = run_id
                all_dfs.append(df_motor)
            except Exception as e:
                logger.error(f"Error loading {csv_path}: {e}")

    if not all_dfs:
        raise ValueError(f"No valid motor CSV files found in {train_path}")

    df_full = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded motor-level dataset with shape {df_full.shape} from {len(run_dirs)} runs.")
    return df_full


def load_robot_level_dataset(
    train_dir: Path | str = RAW_TRAIN_DIR,
    max_runs: int | None = None,
) -> pd.DataFrame:
    """
    Combines the 6 motor CSV files for each test run into a wide robot-level DataFrame.
    Columns: time, data_motor_1_position, ... data_motor_6_voltage,
             data_motor_1_label, ... data_motor_6_label,
             robot_failure (1 if any motor has label=1), test_condition
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Directory not found: {train_path}")

    run_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    if max_runs:
        run_dirs = run_dirs[:max_runs]

    run_dfs = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        combined = pd.DataFrame()
        ref_time = None

        for m in range(1, 7):
            p = run_dir / f"data_motor_{m}.csv"
            if not p.exists():
                continue
            df_m = pd.read_csv(p)
            if ref_time is None and "time" in df_m.columns:
                ref_time = df_m["time"] - df_m["time"].iloc[0]

            cols_to_keep = [c for c in df_m.columns if c != "time"]
            renamed = {c: f"data_motor_{m}_{c}" for c in cols_to_keep}
            df_m_renamed = df_m[cols_to_keep].rename(columns=renamed)
            combined = pd.concat([combined, df_m_renamed], axis=1)

        if ref_time is not None:
            combined.insert(0, "time", ref_time)
        combined["test_condition"] = run_id

        # Derive robot-level failure: 1 if any motor failed
        label_cols = [f"data_motor_{m}_label" for m in range(1, 7) if f"data_motor_{m}_label" in combined.columns]
        if label_cols:
            combined["robot_failure"] = (combined[label_cols].max(axis=1) > 0).astype(int)

        run_dfs.append(combined)

    if not run_dfs:
        raise ValueError("No run directories could be loaded.")

    df_robot = pd.concat(run_dfs, ignore_index=True)
    logger.info(f"Loaded robot-level wide dataset with shape {df_robot.shape}")
    return df_robot


def generate_sample_dataset(
    output_path: Path | str = SAMPLE_DATA_DIR / "synthetic_robot_telemetry.csv",
    n_samples: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic, documented synthetic benchmark dataset based on the
    observed training data distributions for offline educational demonstrations.
    All rows are clearly labeled with source='SYNTHETIC_BENCHMARK'.
    """
    rng = np.random.default_rng(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    time_steps = np.arange(0, n_samples * 0.1, 0.1)[:n_samples]
    motor_ids = rng.choice(ALL_MOTORS, size=n_samples)
    scenarios = rng.choice(["baseline_operation", "pick_and_place", "stress_test"], size=n_samples)

    # Base nominal signals
    positions = rng.uniform(50, 950, size=n_samples)
    temperatures = 25.0 + 15.0 * np.sin(time_steps / 20.0) + rng.normal(0, 1.5, size=n_samples)
    voltages = 7200.0 + rng.normal(0, 80.0, size=n_samples)
    labels = np.zeros(n_samples, dtype=int)

    # Inject realistic fault episodes (~7% prevalence)
    n_faults = int(n_samples * 0.07)
    fault_indices = rng.choice(n_samples, size=n_faults, replace=False)
    for idx in fault_indices:
        labels[idx] = 1
        temperatures[idx] += rng.uniform(8.0, 30.0)  # Thermal rise
        voltages[idx] -= rng.uniform(150.0, 450.0)   # Voltage sag
        if rng.random() < 0.15:
            positions[idx] = rng.choice([-32000.0, 999.0])  # Sensor glitch

    df_sample = pd.DataFrame({
        "time": np.round(time_steps, 3),
        "position": np.round(positions, 1),
        "temperature": np.round(temperatures, 1),
        "voltage": np.round(voltages, 1),
        "label": labels,
        "motor": motor_ids,
        "test_condition": "synthetic_run_01",
        "data_origin": "SYNTHETIC_BENCHMARK_LOCAL",
    })

    df_sample.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic benchmark dataset at {output_path} with {len(df_sample)} rows.")
    return df_sample


def load_sample_dataset(sample_path: Path | str = SAMPLE_DATA_DIR / "synthetic_robot_telemetry.csv") -> pd.DataFrame:
    """Loads the pre-generated synthetic sample benchmark dataset."""
    p = Path(sample_path)
    if not p.exists():
        return generate_sample_dataset(p)
    return pd.read_csv(p)
