"""
Page 6: Interactive Sensor Lab & Real-Time Simulation
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import ARTIFACTS_DIR, PRIMARY_SENSOR_FEATURES
from src.simulation.simulation_engine import SensorSimulationEngine
from src.simulation.operating_scenarios import OPERATING_SCENARIOS
from src.inference.prediction_service import LocalPredictionService
from src.models.model_registry import list_saved_artifacts
from src.ui.page_data_explorer import get_cached_telemetry


def get_simulation_engine() -> SensorSimulationEngine:
    """Retrieves or initializes the session simulation engine."""
    if "sim_engine" not in st.session_state:
        st.session_state["sim_engine"] = SensorSimulationEngine(buffer_size=120)
    return st.session_state["sim_engine"]


def render_page_simulation():
    st.subheader("Interactive Sensor Lab & Live Fault Diagnostics")

    engine = get_simulation_engine()

    # Load deployable prediction service
    artifacts = list_saved_artifacts(ARTIFACTS_DIR)
    prediction_service = None
    if artifacts:
        # Load best or first model
        best_art = artifacts[0]
        prediction_service = LocalPredictionService(
            model_artifact_path=best_art["model_path"],
            decision_threshold=0.5,
        )

    # Mode Selector
    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        sim_mode = st.radio(
            "Select Telemetry Source Mode",
            ["Mode A: Real Kaggle Telemetry Replay", "Mode B: Synthetic Fault Scenarios"],
            index=0 if engine.mode == "REAL_DATA" else 1,
        )

    with c_m2:
        if sim_mode.startswith("Mode A"):
            engine.set_synthetic_mode()
            raw_data = get_cached_telemetry(max_runs=5)
            # Motor & run selector
            sel_run = st.selectbox("Select Test Run to Replay", sorted(list(raw_data["test_condition"].unique())))
            sel_motor = st.selectbox("Select Motor", sorted(list(raw_data["motor"].unique())))
            run_data = raw_data[(raw_data["test_condition"] == sel_run) & (raw_data["motor"] == sel_motor)]
            engine.set_real_data_source(run_data)
            st.caption(f"Replaying {len(run_data)} authentic Kaggle samples from Run '{sel_run}', Motor '{sel_motor}'.")
        else:
            engine.set_synthetic_mode()
            scenario_keys = list(OPERATING_SCENARIOS.keys())
            scenario_names = [OPERATING_SCENARIOS[k]["display_name"] for k in scenario_keys]
            sel_idx = st.selectbox("Select Operating Scenario to Inject", range(len(scenario_keys)), format_func=lambda i: scenario_names[i])
            chosen_scenario_key = scenario_keys[sel_idx]
            st.caption(OPERATING_SCENARIOS[chosen_scenario_key].get("description", ""))

    st.markdown("---")

    # Simulation Action Bar
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    with b1:
        step_clicked = st.button("▶ Step +1", use_container_width=True)
    with b2:
        run_multi = st.button("⏩ Stream +5 Steps", use_container_width=True)
    with b3:
        if st.button("🔄 Reset Buffer", use_container_width=True):
            engine.reset()
            st.rerun()

    # Advance simulation if clicked
    n_steps = 5 if run_multi else (1 if step_clicked else 0)
    latest_sample = None
    for _ in range(n_steps):
        scenario_key = chosen_scenario_key if sim_mode.startswith("Mode B") else "NORMAL_STEADY_STATE"
        latest_sample = engine.step(
            scenario_key=scenario_key,
            prediction_service=prediction_service,
            custom_threshold=0.5,
        )

    # Render Current Status
    hist_df = engine.get_history_dataframe()

    if not hist_df.empty and latest_sample is not None:
        st.markdown("### Real-Time Sensor Telemetry & Diagnostics")

        # Top gauges
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Position (Encoder)", f"{latest_sample.get('position', 0):.1f} steps")
        s2.metric("Temperature", f"{latest_sample.get('temperature', 0):.1f} °C")
        s3.metric("Voltage", f"{latest_sample.get('voltage', 0):.1f} mV")

        pred_class = latest_sample.get("model_predicted_class")
        prob_fault = latest_sample.get("model_fault_probability")

        with s4:
            if pred_class == 1:
                st.markdown("""
                <div class="metric-card" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.15);">
                    <div class="metric-label" style="color: #fca5a5;">AI Diagnosis</div>
                    <div class="metric-value" style="color: #ef4444; font-size: 1.4rem;">⚠️ FAULT DETECTED</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card" style="border-color: #10b981; background: rgba(16, 185, 129, 0.15);">
                    <div class="metric-label" style="color: #86efac;">AI Diagnosis</div>
                    <div class="metric-value" style="color: #10b981; font-size: 1.4rem;">✅ NORMAL</div>
                </div>
                """, unsafe_allow_html=True)

        # Scenario vs Prediction Comparison
        c_ver1, c_ver2 = st.columns(2)
        with c_ver1:
            origin_tag = latest_sample.get("data_origin", "SYNTHETIC")
            st.markdown(f"**Data Origin**: `{origin_tag}`")
            if "scenario_intended_label" in latest_sample:
                intended = latest_sample["scenario_intended_label"]
                intended_str = "Normal (0)" if intended == 0 else ("Fault (1)" if intended == 1 else "Custom")
                st.markdown(f"**Configured Scenario Intention**: `{intended_str}`")
            elif "actual_ground_truth_label" in latest_sample:
                gt = latest_sample["actual_ground_truth_label"]
                st.markdown(f"**Actual Ground Truth in Kaggle Data**: `{'Normal (0)' if gt == 0 else 'Fault (1)'}`")

        with c_ver2:
            prob_str = f"{prob_fault*100:.1f}%" if prob_fault is not None else "N/A"
            st.markdown(f"**Model Fault Probability**: `{prob_str}` (Decision Threshold: 0.50)")
            # Honest agreement check
            if "scenario_intended_label" in latest_sample and latest_sample["scenario_intended_label"] in [0, 1]:
                agreement = pred_class == latest_sample["scenario_intended_label"]
                if agreement:
                    st.markdown("**Status**: :green[Model Prediction Agrees with Configured Scenario]")
                else:
                    st.markdown("**Status**: :orange[Disagreement: Model output differs from configured scenario intention]")

        # Real-time bounded line chart
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=hist_df["time"],
            y=hist_df["temperature"],
            name="Temperature (°C)",
            line=dict(color="#f97316", width=2),
        ))
        fig_live.add_trace(go.Scatter(
            x=hist_df["time"],
            y=hist_df["voltage"] / 100.0,  # Scaled for dual visibility
            name="Voltage (mV / 100)",
            line=dict(color="#38bdf8", width=2),
        ))
        fig_live.update_layout(
            title=f"Bounded Telemetry Stream Buffer (Last {len(hist_df)} Observations)",
            xaxis_title="Simulation Time Elapsed (seconds)",
            yaxis_title="Sensor Magnitude",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_live, use_container_width=True)

        # Local explanation snippet
        expl = latest_sample.get("model_explanation", {})
        if expl.get("explanation_text"):
            st.info(f"**Diagnostic Reasoning**: {expl['explanation_text']}")
    else:
        st.info("Simulation buffer is empty. Click **▶ Step +1** or **⏩ Stream +5 Steps** above to start live streaming.")
