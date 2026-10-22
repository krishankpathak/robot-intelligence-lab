# System Architecture Document

## Overview
The **ROBOT INTELLIGENCE LAB** is structured as a decoupled, multi-tier Python application designed to execute 100% locally on personal computers, laptops, or edge compute hardware.

```
                           +----------------------------+
                           |  Streamlit Multi-Page UI   |
                           +----------------------------+
                                        │
           ┌────────────────────────────┼───────────────────────────┐
           ▼                            ▼                           ▼
+---------------------+     +----------------------+     +----------------------+
| Data Ingestion &    |     | Model Training &     |     | Interactive Sensor   |
| Quality Validation  |     | Evaluation Engine    |     | Simulation Engine    |
+---------------------+     +----------------------+     +----------------------+
           │                            │                           │
           │                            ▼                           ▼
           │                +----------------------+     +----------------------+
           │                | Preprocessing Pipe   |     | Local Prediction     |
           │                | & Model Registry     | ──> | Service & OOD Check  |
           │                +----------------------+     +----------------------+
           ▼                            │                           │
+---------------------+                 ▼                           ▼
| Kaggle Raw Data &   |     +----------------------+     +----------------------+
| Synthetic Cache     |     | Artifact Storage     |     | Deterministic        |
| (Read-Only)         |     | (.joblib + SHA-256)  |     | Attribution Engine   |
+---------------------+     +----------------------+     +----------------------+
```

## Module Responsibilities

### 1. Presentation Layer (`src/ui/`, `app.py`)
- Built using Streamlit and modern vanilla CSS.
- Encapsulates view logic across 9 specialized workflow pages.
- Does not contain hardcoded raw data or fake evaluation metrics; dynamically queries the underlying services.

### 2. Ingestion & Quality Layer (`src/data/`)
- `dataset_loader.py`: Ingests the 6-motor CSV streams per run, reads `Test conditions.xlsx`, and creates documented synthetic fallbacks when required. Raw files remain strictly read-only.
- `data_validation.py`: Enforces schema integrity, data types, and physical sensor range checks.
- `data_quality.py`: Scans for missing entries, duplicates, constant features, and computes class imbalance ratios.

### 3. Preprocessing & Splitting Layer (`src/preprocessing/`)
- `cleaning.py`: Removes exact duplicates and sanitizes extreme electrical dropout spikes (e.g. -32271).
- `transformations.py`: Constructs scikit-learn `ColumnTransformer` pipelines combining median imputation and standard/robust scaling.
- `data_splitting.py`: Enforces **Grouped Run Splitting** across operational sequences to prevent high-frequency temporal data leakage.

### 4. Machine Learning & Registry Layer (`src/models/`, `src/evaluation/`)
- Integrates standard estimators (Dummy Baseline, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting).
- Implements from-scratch algorithms in pure NumPy (`EducationalLogisticRegression`, `EducationalDecisionStump`).
- Manages artifact persistence via `joblib` with SHA-256 cryptographic checksums in `artifacts/`.

### 5. Sensor Simulation Layer (`src/simulation/`)
- `SensorSimulationEngine`: Maintains a bounded `deque(maxlen=150)` buffer.
- Clearly separates Mode A (authentic Kaggle telemetry replay) from Mode B (synthetic scenario injection).
- Guarantees zero leakage: scenario labels are never passed as inputs to the predictive model.
