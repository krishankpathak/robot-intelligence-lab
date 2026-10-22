"""
ROBOT INTELLIGENCE LAB — Main Streamlit Application
Local AI-Powered Predictive Maintenance, Sensor Analytics, and Interactive Simulation
"""
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.styles import render_header
from src.ui.page_overview import render_page_overview
from src.ui.page_data_explorer import render_page_data_explorer
from src.ui.page_eda import render_page_eda
from src.ui.page_training import render_page_training
from src.ui.page_evaluation import render_page_evaluation
from src.ui.page_simulation import render_page_simulation
from src.ui.page_predict import render_page_predict
from src.ui.page_educational import render_page_educational
from src.ui.page_history import render_page_history

# Page configuration
st.set_page_config(
    page_title="Robot Intelligence Lab — Predictive Maintenance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Render top hero banner
    render_header()

    # Sidebar Navigation
    st.sidebar.title("🤖 Navigation")
    pages = {
        "1. Overview & Architecture": render_page_overview,
        "2. Data Explorer & Quality": render_page_data_explorer,
        "3. Exploratory Sensor Analysis": render_page_eda,
        "4. Model Training Lab": render_page_training,
        "5. Model Evaluation & Metrics": render_page_evaluation,
        "6. Live Sensor Lab (Simulation)": render_page_simulation,
        "7. Predict a Sample (OOD Check)": render_page_predict,
        "8. Educational From-Scratch Lab": render_page_educational,
        "9. Experiment History & Provenance": render_page_history,
    }

    selected_page = st.sidebar.radio(
        "Workflow Steps",
        list(pages.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Environment & Policy:**
    - Stack: Python 3.11 • scikit-learn • Streamlit
    - Mode: **100% Local AI**
    - External APIs: **Zero (Forbidden)**
    - Dataset: **Kaggle Robot Telemetry**
    """)

    # Route to selected page
    page_func = pages[selected_page]
    page_func()


if __name__ == "__main__":
    main()
