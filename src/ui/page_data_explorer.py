"""
Page 2: Data Explorer & Schema Inspector
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.data.dataset_loader import load_motor_level_dataset, load_robot_level_dataset, load_sample_dataset
from src.data.data_quality import run_full_quality_audit
from src.analysis.statistical_analysis import compute_univariate_summary


@st.cache_data(show_spinner="Loading dataset...")
def get_cached_telemetry(max_runs: int = 10):
    try:
        return load_motor_level_dataset(max_runs=max_runs)
    except Exception:
        return load_sample_dataset()


def render_page_data_explorer():
    st.subheader("Data Explorer & Quality Audit")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        st.markdown(
            "Inspect raw robot telemetry sequences loaded from the Kaggle predictive-maintenance dataset. "
            "Telemetry is sampled at 10 Hz across 6 motors."
        )
    with col_ctrl2:
        max_runs = st.selectbox("Number of Runs to Load into Memory", [5, 10, 23], index=1)

    df = get_cached_telemetry(max_runs=max_runs)

    tab1, tab2, tab3 = st.tabs(["Raw Telemetry Table", "Schema & Quality Audit", "Univariate Descriptive Statistics"])

    with tab1:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            motors = ["All"] + sorted(list(df["motor"].unique()))
            selected_motor = st.selectbox("Filter by Motor", motors)
        with f2:
            runs = ["All"] + sorted(list(df["test_condition"].unique()))
            selected_run = st.selectbox("Filter by Run Sequence", runs)
        with f3:
            labels = ["All", "Normal Only (0)", "Faults Only (1)"]
            selected_label = st.selectbox("Filter by Health State", labels)

        filtered_df = df.copy()
        if selected_motor != "All":
            filtered_df = filtered_df[filtered_df["motor"] == selected_motor]
        if selected_run != "All":
            filtered_df = filtered_df[filtered_df["test_condition"] == selected_run]
        if selected_label == "Normal Only (0)":
            filtered_df = filtered_df[filtered_df["label"] == 0]
        elif selected_label == "Faults Only (1)":
            filtered_df = filtered_df[filtered_df["label"] == 1]

        st.caption(f"Displaying {len(filtered_df):,} matching observations (out of {len(df):,} loaded):")
        st.dataframe(filtered_df.head(100), use_container_width=True)

    with tab2:
        audit = run_full_quality_audit(df, target_col="label")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Records", f"{audit['total_records']:,}")
        q2.metric("Missing Values", "0 (None)" if not audit["has_missing_values"] else "Detected")
        q3.metric("Duplicate Rows", f"{audit['duplicate_rows']}")
        imb = audit["imbalance_report"]
        q4.metric("Class Imbalance Ratio", f"{imb.get('imbalance_ratio', 1.0)} : 1")

        st.markdown("#### Column Schema Details")
        schema_df = pd.DataFrame({
            "Column Name": list(df.columns),
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": [df[c].count() for c in df.columns],
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Example Value": [str(df[c].iloc[0]) for c in df.columns],
        })
        st.dataframe(schema_df, use_container_width=True)

        st.markdown("#### Outlier Counts (1.5 * IQR Rule)")
        outliers = audit.get("outlier_counts", {})
        outlier_df = pd.DataFrame([
            {"Feature": k, "Outlier Count": v, "Percentage": f"{(v / len(df))*100:.2f}%"}
            for k, v in outliers.items()
        ])
        st.dataframe(outlier_df, use_container_width=True)

    with tab3:
        st.markdown("#### Descriptive Statistics for Numeric Features")
        numeric_feats = ["position", "temperature", "voltage"]
        stats_df = compute_univariate_summary(df, numeric_feats)
        st.dataframe(stats_df, use_container_width=True)
