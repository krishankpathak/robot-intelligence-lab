"""
Page 7: Single-Sample Inference & Out-of-Distribution Diagnostics
"""
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ARTIFACTS_DIR, SENSOR_BOUNDS
from src.models.model_registry import list_saved_artifacts
from src.inference.prediction_service import LocalPredictionService
from src.inference.confidence_analysis import analyze_prediction_confidence


def render_page_predict():
    st.subheader("Single-Observation Telemetry Inference & OOD Diagnostics")

    st.markdown(
        "Manually configure robot sensor observations to test the model's response. "
        "The service validates input schemas, estimates fault probability, and identifies out-of-distribution inputs."
    )

    artifacts = list_saved_artifacts(ARTIFACTS_DIR)
    if not artifacts:
        st.warning("No saved model artifacts found. Please train models first.")
        return

    model_options = [a["model_name"] for a in artifacts]
    selected_name = st.selectbox("Active Inference Model", model_options)
    matching = [a for a in artifacts if a["model_name"] == selected_name][0]

    baseline_stats = {
        "position": {"mean": 414.5, "std": 261.4, "min": 0.0, "max": 999.0},
        "temperature": {"mean": 36.6, "std": 10.0, "min": 21.0, "max": 65.0},
        "voltage": {"mean": 7189.1, "std": 230.4, "min": 6800.0, "max": 7600.0},
    }

    prediction_service = LocalPredictionService(
        model_artifact_path=matching["model_path"],
        training_baseline_stats=baseline_stats,
        decision_threshold=0.5,
    )

    st.markdown("#### Configure Telemetry Sensor Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        pos_val = st.number_input(
            "Position (steps)",
            min_value=-35000.0,
            max_value=2000.0,
            value=450.0,
            step=10.0,
            help="Nominal encoder range: 0 - 999 steps",
        )
    with c2:
        temp_val = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=250.0,
            value=35.0,
            step=1.0,
            help="Nominal coil temperature: 21 - 50 °C. Overheating occurs above 60 °C.",
        )
    with c3:
        volt_val = st.number_input(
            "Voltage (mV)",
            min_value=-30000.0,
            max_value=12000.0,
            value=7200.0,
            step=50.0,
            help="Nominal voltage across motor: ~7200 mV (7.2 V). Drops below 6900 under heavy load.",
        )

    threshold_val = st.slider("Classification Decision Threshold", 0.05, 0.95, 0.50, 0.05)

    if st.button("🔍 Run Local AI Diagnostics", type="primary"):
        sample_dict = {"position": pos_val, "temperature": temp_val, "voltage": volt_val}
        res = prediction_service.predict_sample(sample_dict, custom_threshold=threshold_val)

        if not res["success"]:
            st.error(f"Inference Validation Error: {res['error']}")
            return

        pred_class = res["predicted_class"]
        prob = res["fault_probability"]
        expl = res["explanation"]

        st.markdown("### AI Diagnostic Decision")
        r1, r2, r3 = st.columns(3)
        with r1:
            if pred_class == 1:
                st.error("🚨 FAULT DETECTED (Class 1)")
            else:
                st.success("✅ NORMAL OPERATION (Class 0)")

        with r2:
            prob_pct = f"{prob*100:.1f}%" if prob is not None else "N/A"
            st.metric("Fault Probability", prob_pct)

        with r3:
            if prob is not None:
                conf = analyze_prediction_confidence(prob, threshold=threshold_val)
                st.metric("Decision Confidence", conf["confidence_level"])

        # OOD Warning
        if expl.get("is_out_of_distribution"):
            for w in expl.get("ood_warnings", []):
                st.markdown(f"""
                <div class="alert-box alert-warning">
                    ⚠️ <strong>Out-of-Distribution Sensor Alert</strong>: {w}<br>
                    Predictions on readings beyond observed empirical training ranges cannot be guaranteed reliable.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Deterministic Feature Attribution & Explanation")
        st.info(expl.get("explanation_text", ""))

        st.markdown("#### Sensor Deviations from Training Population Baselines")
        dev_df = pd.DataFrame(expl.get("deviations", []))
        st.dataframe(dev_df, use_container_width=True)
        st.caption(expl.get("caveat", ""))
