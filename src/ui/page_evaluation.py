"""
Page 5: Model Evaluation, Confusion Matrix & Feature Importance
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import ARTIFACTS_DIR, PRIMARY_SENSOR_FEATURES
from src.models.model_registry import list_saved_artifacts, load_model_artifact
from src.evaluation.classification_metrics import (
    evaluate_at_thresholds,
    compute_roc_pr_curves,
)
from src.explainability.feature_importance import extract_tree_importance, CAVEAT_EXPLANATION


def render_page_evaluation():
    st.subheader("Model Evaluation, Diagnostics & Explainability")

    artifacts = list_saved_artifacts(ARTIFACTS_DIR)
    if not artifacts:
        st.warning("No saved model artifacts found. Please train models in the 'Model Training' tab first.")
        return

    model_options = [a["model_name"] for a in artifacts]
    selected_name = st.selectbox("Select Model to Evaluate", model_options)

    # Find artifact path
    matching = [a for a in artifacts if a["model_name"] == selected_name][0]
    pipeline, meta = load_model_artifact(matching["model_path"])
    metrics = meta.get("metrics", {})

    st.markdown("#### Key Evaluation Scorecards")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F1-Score", f"{metrics.get('f1', 0.0):.4f}")
    m2.metric("Recall (Fault Detection)", f"{metrics.get('recall', 0.0):.4f}")
    m3.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
    m4.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0.0):.4f}")
    m5.metric("PR-AUC", f"{metrics.get('pr_auc', 0.0) or 0.0:.4f}")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix Diagnostics", "Decision Threshold Tuner", "Feature Importance"])

    with tab1:
        st.markdown("#### Confusion Matrix Heatmap")
        cm = metrics.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        matrix_data = [
            [cm.get("tn", 0), cm.get("fp", 0)],
            [cm.get("fn", 0), cm.get("tp", 0)],
        ]
        fig_cm = px.imshow(
            matrix_data,
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Observations"),
            x=["Normal (0)", "Fault (1)"],
            y=["Normal (0)", "Fault (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected_name}",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        c_fn, c_fp = st.columns(2)
        with c_fn:
            st.error(f"**False Negatives (Missed Faults)**: {cm.get('fn', 0):,} observations where robot was faulty but predicted normal.")
        with c_fp:
            st.warning(f"**False Positives (False Alarms)**: {cm.get('fp', 0):,} observations where normal robot triggered alarm.")

    with tab2:
        st.markdown(
            "#### Probability Decision Threshold Optimizer\n"
            "Adjusting the decision threshold allows maintenance managers to balance "
            "missed faults (costly breakdown) against false alarms (unnecessary maintenance stop)."
        )
        th_val = st.slider("Select Custom Decision Threshold", 0.05, 0.95, 0.50, 0.05)

        st.caption(
            f"At threshold = {th_val:.2f}: If model's estimated fault probability >= {th_val:.2f}, "
            "it triggers an alert; otherwise it classifies as normal."
        )

    with tab3:
        st.markdown("#### Feature Importance Analysis")
        st.caption(CAVEAT_EXPLANATION)

        df_imp = extract_tree_importance(pipeline, PRIMARY_SENSOR_FEATURES)
        fig_imp = px.bar(
            df_imp,
            x="importance_percent",
            y="feature",
            orientation="h",
            color="importance_percent",
            color_continuous_scale="Viridis",
            title=f"MDI Feature Importance Breakdown (%) — {selected_name}",
            labels={"importance_percent": "Relative Contribution (%)", "feature": "Sensor Feature"},
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
