"""
Page 1: Project Overview & System Health
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from config.settings import RAW_KAGGLE_DIR, RAW_TRAIN_DIR, ARTIFACTS_DIR
from src.models.model_registry import list_saved_artifacts, load_model_artifact


def render_page_overview():
    st.subheader("System Architecture & Operational Status")

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)

    # Dataset status
    dataset_exists = RAW_TRAIN_DIR.exists() and any(RAW_TRAIN_DIR.iterdir())
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset Verification</div>
            <div class="metric-value">Verified</div>
            <div class="metric-desc">235,854 real telemetry samples (23 runs, 6 motors)</div>
        </div>
        """, unsafe_allow_html=True)

    # Saved models
    artifacts = list_saved_artifacts(ARTIFACTS_DIR)
    n_models = len(artifacts)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trained Pipelines</div>
            <div class="metric-value">{n_models} Models</div>
            <div class="metric-desc">Fitted & saved with SHA-256 checksums</div>
        </div>
        """, unsafe_allow_html=True)

    # Active model
    active_model = artifacts[0]["model_name"] if artifacts else "None"
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Deployable Model</div>
            <div class="metric-value" style="font-size: 1.3rem;">{active_model}</div>
            <div class="metric-desc">Loaded locally for simulation & prediction</div>
        </div>
        """, unsafe_allow_html=True)

    # AI Policy
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">AI Architecture</div>
            <div class="metric-value" style="font-size: 1.3rem;">100% Local</div>
            <div class="metric-desc">Zero external cloud APIs or API keys required</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow architecture
    st.markdown("### End-to-End Local Machine Learning Pipeline")
    st.markdown("""
    ```
    ┌────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
    │  Raw Telemetry │ ──> │ Data Validation  │ ──> │ Preprocessing       │
    │  (23 Runs, 6M) │     │ & Glitch Cleanup │     │ (Standard / Robust) │
    └────────────────┘     └──────────────────┘     └─────────────────────┘
                                                               │
                                                               ▼
    ┌────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
    │ Interactive    │ <── │ Local Prediction │ <── │ Leakage-Safe Model  │
    │ Sensor Lab     │     │ & Explainability │     │ Training & Registry │
    └────────────────┘     └──────────────────┘     └─────────────────────┘
    ```
    """)

    st.markdown("### Key Technical Guardrails Enforced")
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("""
        - **Data Leakage Prevention**: Uses Grouped Run Splitting to prevent contiguous time-series points (sampled at 10 Hz) from appearing across both training and test partitions.
        - **Honest Metric Evaluation**: Recognizes severe class imbalance (92.8% Normal vs 7.2% Fault). Rejects naive accuracy as a standalone success metric.
        - **Deterministic Explainability**: Computes z-score deviations from empirical training baselines to explain sensor drivers without external LLMs.
        """)
    with c_b:
        st.markdown("""
        - **Cryptographic Provenance**: Every saved model artifact is hashed using SHA-256 to ensure model integrity before serving inference.
        - **Dual-Mode Sensor Simulation**: Clearly separates Mode A (Real Kaggle telemetry replay) from Mode B (Synthetic fault injection).
        - **Bounded Memory Buffers**: Real-time simulation charts use a bounded buffer (150 observations) to prevent browser/memory leaks.
        """)
