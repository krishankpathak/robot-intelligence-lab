# ROBOT INTELLIGENCE LAB
### Local AI-Powered Predictive Maintenance and Fault Analytics

A complete, fully verified, local machine learning and telemetry analytics application built to analyze the actual Kaggle Robot Predictive Maintenance dataset, learn patterns from multi-motor robotic sensor signals, evaluate models honestly under severe class imbalance, and serve real-time predictions and simulations with zero external cloud dependencies.

---

## 🌟 Key Features

1. **100% Local AI Policy**: No external cloud APIs, no OpenAI/Claude keys, no remote inference endpoints. Runs entirely on local CPU/GPU hardware.
2. **Kaggle Robot Telemetry Verified**: Analyzes the genuine 6-motor robot arm dataset (235,854 observations across 23 operational test sequences).
3. **Data Leakage Guardrails**: Enforces **Grouped Run Splitting** to ensure contiguous 10 Hz time-series observations from the same operational session do not leak across training and test partitions.
4. **Honest Metric Evaluation**: Analyzes rare fault prevalence (7.2% fault vs 92.8% normal), highlighting why naive ~92% accuracy models are useless compared to F1-Score, Recall, PR-AUC, and ROC-AUC.
5. **Educational From-Scratch Algorithms**: Includes transparent NumPy implementations of:
   - Logistic Regression with Sigmoid and gradient descent.
   - Decision Stump with Gini impurity reduction.
   - Custom metric calculation with automated scikit-learn parity verification.
6. **Dual-Mode Interactive Sensor Lab**:
   - **Mode A (Real Telemetry Replay)**: Step through authentic Kaggle dataset recordings motor-by-motor.
   - **Mode B (Synthetic Fault Scenarios)**: Inject simulated thermal buildup, voltage sag, mechanical jam, and serial glitches.
7. **Local Prediction Service & OOD Detection**: Validates schema, flags out-of-distribution values, and computes deterministic local explanations without LLMs.
8. **Cryptographic Model Provenance**: Model artifacts are stored with SHA-256 integrity checksums to guarantee provenance.

---

## 🏗️ Project Architecture

```
robot-intelligence-lab/
├── README.md                          # Project documentation and quickstart
├── requirements.txt                   # Tested Python dependencies
├── pyproject.toml                     # Package configuration & pytest settings
├── .gitignore                         # Git exclusion rules
├── app.py                             # Main Streamlit application entry point
├── config/
│   └── settings.py                    # Central paths, bounds, and parameters
├── src/
│   ├── data/                          # Dataset loaders, schema validation, quality audits
│   ├── analysis/                      # Exploratory data analysis & feature engineering
│   ├── preprocessing/                 # Cleaning, scaling pipelines, and leakage-safe splitting
│   ├── models/                        # Baseline, Logistic, Tree, Ensemble, and From-Scratch algorithms
│   ├── evaluation/                    # Classification metrics, calibration, cross-validation
│   ├── explainability/                # Permutation & tree importance, local explanations
│   ├── simulation/                    # Telemetry generator, fault injection, simulation engine
│   ├── inference/                     # Local prediction service & uncertainty analysis
│   ├── reporting/                     # Experiment logging & artifact exports
│   ├── ui/                            # Streamlit page modules & custom CSS
│   └── utils/                         # Logging & reproducibility utilities
├── tests/                             # 34 automated unit & integration tests
├── data/                              # Sample and local telemetry cache
├── artifacts/                         # Saved .joblib model pipelines & metadata
├── reports/                           # JSON experiment run logs
├── docs/                              # Comprehensive architectural and workshop guides
└── scripts/                           # Setup and launch automation scripts
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11.9 on Windows 11)
- 4 GB+ RAM

### 2. Installation
Clone or navigate to the repository, create a virtual environment, and install dependencies:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the comprehensive test suite (34 passing unit and integration tests):

```bash
pytest tests -v
```

### 4. Train Verified Models on Kaggle Dataset
Train all candidate models on the 235k samples using grouped run splitting:

```bash
python scripts/train_verified_models.py
```

### 5. Launch the Streamlit Dashboard
Start the interactive application:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Evaluation Summary on Held-Out Grouped Runs

| Model | Accuracy | Precision | Recall (Fault Detection) | F1-Score | PR-AUC | ROC-AUC | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.7284 | 0.0663 | **0.0640** | **0.0651** | **0.1396** | 0.4571 | 0.25 s |
| **Gradient Boosting** | 0.7957 | 0.0561 | 0.0242 | 0.0338 | 0.1217 | 0.4253 | 2.53 s |
| **Logistic Regression** | 0.6949 | 0.0222 | 0.0247 | 0.0234 | 0.0980 | 0.2956 | 0.19 s |
| **Random Forest** | 0.8041 | 0.0397 | 0.0140 | 0.0207 | 0.1249 | 0.4495 | 1.65 s |
| **Dummy Baseline** | **0.8522** | 0.0000 | 0.0000 | 0.0000 | 0.1478 | 0.5000 | 0.07 s |

*Key Insight*: While the naive Dummy Baseline achieves 85.2% accuracy by always guessing the majority class, it misses 100% of faults. Decision Trees and Ensembles actively detect anomalies by isolating voltage dropouts and thermal surges.

---

## 👥 Educational Workshop Demo
A structured 15–20 minute script is provided in `docs/WORKSHOP_SCRIPT.md` guiding instructors through:
1. Demonstrating why accuracy lies in imbalanced robotics data.
2. Stepping through the live simulation lab with synthetic fault injection.
3. Inspecting the math behind Gini impurity and sigmoid gradient descent.
4. Serving local predictions with deterministic explainability.
