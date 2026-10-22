"""
Page 3: Exploratory Data Analysis & Sensor Visualizations
"""
import streamlit as st
import pandas as pd

from src.ui.page_data_explorer import get_cached_telemetry
from src.analysis.exploratory_analysis import (
    plot_target_distribution,
    plot_sensor_histogram,
    plot_sensor_boxplots,
    plot_correlation_heatmap,
    plot_telemetry_trajectory,
)
from src.analysis.statistical_analysis import compute_class_stratified_stats, compute_correlation_matrix


def render_page_eda():
    st.subheader("Exploratory Data Analysis & Sensor Dynamics")
    df = get_cached_telemetry(max_runs=10)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Imbalance Analysis",
        "Sensor Distributions by Health State",
        "Correlation & Statistical Drift",
        "Time-Series Run Trajectory",
    ])

    with tab1:
        st.markdown(
            "#### Target Distribution & Rarity of Failures\n"
            "Predictive maintenance datasets are inherently characterized by extreme class imbalance: "
            "normal operation accounts for over 92% of all samples, while real mechanical/electrical faults occur rarely (~7%)."
        )
        c_pie, c_info = st.columns([3, 2])
        with c_pie:
            fig_pie = plot_target_distribution(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_info:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>Why Accuracy is Misleading Here:</strong><br>
                A naive "Dummy Model" that always predicts Normal (Class 0) achieves ~92% accuracy while detecting 
                <strong>zero</strong> actual faults (0.0% Recall). In robotics, an undetected motor jam or overheating failure 
                leads to costly mechanical damage and line shutdown.
            </div>
            """, unsafe_allow_html=True)
            strat_stats = compute_class_stratified_stats(df, ["position", "temperature", "voltage"])
            st.dataframe(strat_stats, use_container_width=True)

    with tab2:
        st.markdown("#### Univariate Sensor Distributions Stratified by Class")
        selected_feat = st.selectbox("Select Sensor Feature to Inspect", ["temperature", "voltage", "position"])

        fig_hist = plot_sensor_histogram(df, feature=selected_feat)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Dispersion and Outlier Analysis (Boxplots)")
        fig_box = plot_sensor_boxplots(df, features=["position", "temperature", "voltage"])
        st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.markdown("#### Pairwise Pearson Correlation Heatmap")
        corr_matrix = compute_correlation_matrix(df, ["position", "temperature", "voltage", "label"])
        fig_corr = plot_correlation_heatmap(corr_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            "Note: Correlation reflects linear association. Non-linear relationships or multi-sensor "
            "interactions (e.g. high position displacement combined with thermal surge) are captured by non-linear decision trees."
        )

    with tab4:
        st.markdown("#### Continuous Sensor Run Trajectory")
        runs = sorted(list(df["test_condition"].unique()))
        selected_run = st.selectbox("Select Test Run Sequence", runs, index=min(2, len(runs) - 1))

        df_run = df[df["test_condition"] == selected_run]
        fig_traj = plot_telemetry_trajectory(df_run, features=["temperature", "voltage"])
        st.plotly_chart(fig_traj, use_container_width=True)
