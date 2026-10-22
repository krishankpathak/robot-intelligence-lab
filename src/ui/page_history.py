"""
Page 9: Experiment History, Artifact Provenance & Export
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import json

from config.settings import REPORTS_DIR, ARTIFACTS_DIR
from src.reporting.experiment_report import load_all_experiment_reports
from src.models.model_registry import list_saved_artifacts, compute_file_sha256


def render_page_history():
    st.subheader("Experiment History & Cryptographic Provenance")

    st.markdown(
        "Review historical training runs, hyperparameters, validation scores, and verify "
        "the cryptographic integrity of persisted model artifacts."
    )

    tab1, tab2 = st.tabs(["Experiment Run History", "Saved Artifacts & Cryptographic Signatures"])

    with tab1:
        st.markdown("#### Logged Experiment Runs")
        df_exp = load_all_experiment_reports(REPORTS_DIR)

        if not df_exp.empty:
            st.dataframe(df_exp, use_container_width=True)

            # JSON detail viewer
            st.markdown("#### Detailed Experiment JSON Viewer")
            run_ids = list(df_exp["run_id"])
            sel_run = st.selectbox("Select Run to Inspect Raw Configuration", run_ids)

            json_file = REPORTS_DIR / f"{sel_run}.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as jf:
                    st.json(json.load(jf))
        else:
            st.info("No experiment logs found yet. Execute a training run in the 'Model Training' page.")

    with tab2:
        st.markdown("#### Persisted Model Artifacts & SHA-256 Checksums")
        artifacts = list_saved_artifacts(ARTIFACTS_DIR)

        if artifacts:
            rows = []
            for art in artifacts:
                p = Path(art["model_path"])
                sha = compute_file_sha256(p)
                size_kb = np.round(p.stat().st_size / 1024, 1) if p.exists() else 0.0
                rows.append({
                    "Model Name": art["model_name"],
                    "Artifact File": p.name,
                    "File Size (KB)": size_kb,
                    "SHA-256 Hash Signature": sha,
                    "F1 Score": art.get("f1_score"),
                    "Recall": art.get("recall"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Cryptographic verification ensures no model artifact is tampered with prior to deployment.")
        else:
            st.info("No artifacts currently saved in artifacts directory.")
