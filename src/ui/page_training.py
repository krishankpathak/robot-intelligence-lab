"""
Page 4: Model Training & Experiment Execution
"""
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import PRIMARY_SENSOR_FEATURES, ARTIFACTS_DIR, REPORTS_DIR
from src.ui.page_data_explorer import get_cached_telemetry
from src.preprocessing.cleaning import clean_telemetry_data
from src.preprocessing.data_splitting import split_telemetry_data
from src.models.model_training import train_all_models, train_single_model, SUPPORTED_MODEL_TYPES
from src.evaluation.classification_metrics import build_comparison_leaderboard
from src.models.model_registry import save_model_artifact
from src.reporting.experiment_report import log_experiment_run


def render_page_training():
    st.subheader("Model Training & Benchmarking Lab")

    st.markdown(
        "Train, validate, and compare candidate machine learning models locally on robot sensor telemetry. "
        "Strict preprocessing isolation ensures zero data leakage."
    )

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        split_strat = st.selectbox(
            "Data Partition Strategy",
            ["grouped", "temporal", "stratified"],
            index=0,
            help=(
                "Grouped: Holds out entire operational runs (Best practice for time series).\n"
                "Temporal: Uses earlier runs for training and later runs for testing.\n"
                "Stratified: Random sample shuffling (demonstrates temporal leakage risks)."
            ),
        )
        test_size = st.slider("Test Partition Ratio", 0.15, 0.35, 0.25, 0.05)
        scaler_choice = st.selectbox("Feature Scaler", ["standard", "robust", "minmax"], index=0)

    with c_cfg2:
        model_choice = st.selectbox(
            "Estimator Selection",
            ["All Candidate Models (Benchmark Leaderboard)"] + SUPPORTED_MODEL_TYPES,
        )
        random_seed = st.number_input("Reproducibility Random Seed", min_value=1, max_value=9999, value=42)
        handle_glitches = st.checkbox("Sanitize extreme sensor glitches (-32k spikes)", value=True)

    if split_strat == "stratified":
        st.markdown("""
        <div class="alert-box alert-warning">
            ⚠️ <strong>Data Leakage Alert</strong>: You selected Random Stratified Splitting.
            Because robot sensor readings are sampled at 10 Hz, contiguous time points in the same fault event
            will be split across train and test partitions, resulting in inflated performance claims.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box alert-success">
            🛡️ <strong>Leakage Guardrail Active</strong>: Partitions are strictly isolated by operational run.
            Models are evaluated only on completely unseen robot operating sequences.
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Start Local Model Training", type="primary"):
        with st.spinner("Processing telemetry and training local estimators..."):
            raw_df = get_cached_telemetry(max_runs=10)
            clean_df, clean_rep = clean_telemetry_data(raw_df, handle_glitches=handle_glitches)

            split_res = split_telemetry_data(
                clean_df,
                strategy=split_strat,
                test_size=test_size,
                random_state=random_seed,
            )

            train_df = split_res["train_df"]
            test_df = split_res["test_df"]

            if model_choice == "All Candidate Models (Benchmark Leaderboard)":
                results = train_all_models(
                    train_df=train_df,
                    test_df=test_df,
                    feature_names=PRIMARY_SENSOR_FEATURES,
                    scaler_type=scaler_choice,
                    random_state=random_seed,
                )
            else:
                res = train_single_model(
                    model_name=model_choice,
                    X_train=train_df[PRIMARY_SENSOR_FEATURES],
                    y_train=train_df["label"].to_numpy(),
                    X_test=test_df[PRIMARY_SENSOR_FEATURES],
                    y_test=test_df["label"].to_numpy(),
                    feature_names=PRIMARY_SENSOR_FEATURES,
                    scaler_type=scaler_choice,
                    random_state=random_seed,
                )
                results = {model_choice: res}

            # Save artifacts
            for m_name, m_res in results.items():
                save_model_artifact(
                    pipeline=m_res["pipeline"],
                    model_name=m_name,
                    feature_names=PRIMARY_SENSOR_FEATURES,
                    metrics=m_res["metrics"],
                    save_dir=ARTIFACTS_DIR,
                )

            # Store in session state for evaluation page
            st.session_state["latest_training_results"] = results
            st.session_state["latest_split_info"] = split_res

            leaderboard = build_comparison_leaderboard({k: v["metrics"] for k, v in results.items()})

            st.success("Training and local evaluation successfully completed!")
            st.markdown("### Evaluation Leaderboard on Held-Out Test Partition")
            st.dataframe(leaderboard, use_container_width=True)

            best_model = leaderboard.iloc[0]["Model"]
            st.info(f"Top performing model for fault detection: **{best_model}** (Saved to artifacts directory)")
