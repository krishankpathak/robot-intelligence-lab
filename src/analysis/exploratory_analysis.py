"""
Exploratory Data Analysis and Visualization Module for ROBOT INTELLIGENCE LAB
Generates interactive, accessible Plotly charts with units, descriptions, and educational context.
"""
from typing import Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import SENSOR_BOUNDS, TARGET_COLUMN, MOTOR_COLUMN, TIMESTAMP_COLUMN

# Clean color palette
COLOR_NORMAL = "#2ecc71"   # Emerald Green
COLOR_FAULT = "#e74c3c"    # Crimson Red
COLOR_ACCENT = "#3498db"   # Blue
COLOR_BG = "#1e293b"


def plot_target_distribution(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> go.Figure:
    """Creates a donut chart illustrating class frequencies and imbalance."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Label"] = counts["Class"].map({0: "Normal (Class 0)", 1: "Fault (Class 1)"}).fillna("Unknown")

    fig = px.pie(
        counts,
        names="Label",
        values="Count",
        color="Label",
        color_discrete_map={
            "Normal (Class 0)": COLOR_NORMAL,
            "Fault (Class 1)": COLOR_FAULT,
        },
        hole=0.45,
        title="Robot Health Target Distribution (Class Balance & Imbalance Ratio)",
    )
    fig.update_traces(textinfo="percent+value", pull=[0, 0.1])
    fig.update_layout(
        legend_title="Operational State",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_sensor_histogram(
    df: pd.DataFrame,
    feature: str,
    target_col: str = TARGET_COLUMN,
    nbins: int = 40,
) -> go.Figure:
    """Creates stratified histograms comparing sensor readings between Normal and Fault."""
    unit = SENSOR_BOUNDS.get(feature, {}).get("unit", "")
    title = f"Distribution of '{feature}' ({unit}) Stratified by Health State"

    fig = px.histogram(
        df,
        x=feature,
        color=target_col,
        barmode="overlay",
        nbins=nbins,
        color_discrete_map={0: COLOR_NORMAL, 1: COLOR_FAULT},
        labels={target_col: "Health State", feature: f"{feature.title()} [{unit}]"},
        title=title,
        opacity=0.7,
    )
    fig.update_layout(
        xaxis_title=f"{feature.title()} ({unit})",
        yaxis_title="Observation Count",
        legend_title="State (0=Normal, 1=Fault)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_sensor_boxplots(df: pd.DataFrame, features: list[str], target_col: str = TARGET_COLUMN) -> go.Figure:
    """Creates side-by-side boxplots showing median, IQR, and extreme anomaly values."""
    fig = go.Figure()
    for feat in features:
        unit = SENSOR_BOUNDS.get(feat, {}).get("unit", "")
        # Normal
        fig.add_trace(go.Box(
            y=df[df[target_col] == 0][feat],
            name=f"{feat} (Normal)",
            marker_color=COLOR_NORMAL,
            boxpoints="outliers",
        ))
        # Fault
        fig.add_trace(go.Box(
            y=df[df[target_col] == 1][feat],
            name=f"{feat} (Fault)",
            marker_color=COLOR_FAULT,
            boxpoints="outliers",
        ))

    fig.update_layout(
        title="Sensor Dispersion and Outlier Analysis (Normal vs Fault)",
        yaxis_title="Sensor Value",
        boxmode="group",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Creates a correlation matrix heatmap with text annotations."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Pairwise Pearson Correlation Matrix (Sensors & Target)",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_telemetry_trajectory(
    df_run: pd.DataFrame,
    features: list[str] = ["temperature", "voltage"],
    time_col: str = TIMESTAMP_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> go.Figure:
    """Plots continuous telemetry trajectory over time highlighting fault intervals."""
    fig = go.Figure()

    t = df_run[time_col] if time_col in df_run.columns else np.arange(len(df_run)) * 0.1

    for feat in features:
        unit = SENSOR_BOUNDS.get(feat, {}).get("unit", "")
        fig.add_trace(go.Scatter(
            x=t,
            y=df_run[feat],
            mode="lines",
            name=f"{feat} ({unit})",
        ))

    # Highlight fault regions
    if target_col in df_run.columns and df_run[target_col].sum() > 0:
        fault_mask = df_run[target_col] == 1
        fault_t = t[fault_mask]
        if len(fault_t) > 0:
            fig.add_trace(go.Scatter(
                x=fault_t,
                y=df_run.loc[fault_mask, features[0]],
                mode="markers",
                marker=dict(color=COLOR_FAULT, size=6, symbol="x"),
                name="Injected Fault State (1)",
            ))

    fig.update_layout(
        title="Time-Series Sensor Trajectory and Fault Episodes",
        xaxis_title="Time Elapsed (seconds)",
        yaxis_title="Sensor Reading",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
