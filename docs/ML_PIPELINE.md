# Machine Learning Pipeline Document

## 1. Data Ingestion & Sanitization
Sensor telemetry is captured at 10 Hz across 6 motors of an articulated robotic arm.
Raw measurements:
- `position`: Encoder steps (0 to 999).
- `temperature`: Coil temperature (°C).
- `voltage`: Driver voltage (mV).
- `label`: Ground truth operational state (0 = Normal, 1 = Fault).

Extreme glitch filtering: Serial communication corruption causes occasional negative drops (e.g. -32271 in position, -25689 in voltage). These dropouts are treated and clipped to valid median boundaries in `cleaning.py`.

## 2. Preventing Data Leakage
Data leakage occurs when information outside the training dataset is inadvertently used to train a model, resulting in overly optimistic evaluation.

### Why Random Splitting Fails in Robotics:
Telemetry points sampled 100 ms apart are highly auto-correlated. If observations are shuffled randomly:
- Point $t$ goes to the training set.
- Point $t+0.1s$ goes to the test set.
Because the motor position and temperature barely change in 100 ms, the test set simply memorizes adjacent points rather than learning generalizable fault dynamics.

### Implemented Splitting Solutions:
1. **Grouped Run Splitting (Primary)**:
   - Partitions data strictly by `test_condition` (the operational run).
   - Entire runs (e.g., Run `20240426_140055`) are held out.
   - The test set evaluates how the model performs on a robot operating session it has never seen before.
2. **Temporal Splitting**:
   - Earlier calendar runs train the model; latest runs evaluate it.
3. **Strict Preprocessing Fit**:
   - Imputers and scalers are fitted **only** on the training partition:
     ```python
     preprocessor.fit(X_train)
     X_train_scaled = preprocessor.transform(X_train)
     X_test_scaled = preprocessor.transform(X_test)
     ```

## 3. Class Imbalance Strategy
Faults comprise only 7.2% of the dataset.
- **Naive Baseline**: Predicting all zeros yields 85-92% accuracy, but detects 0% of faults (F1 = 0.0).
- **Class Weighting**: Estimators use `class_weight='balanced'` or `class_weight='balanced_subsample'` to assign proportionally higher loss penalties to false negatives on minority fault instances:
  $$w_1 = \frac{N}{2 \cdot N_1}$$
- **Threshold Optimization**: Rather than fixing a naive 0.5 probability threshold, the application allows operators to calibrate decision thresholds (e.g., 0.35 - 0.45) to prioritize high recall and prevent catastrophic physical downtime.

## 4. Pipeline Packaging & Provenance
Fitted transformers and estimators are bundled together in a single scikit-learn `Pipeline`:
```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", estimator)
])
```
This ensures identical transformations are applied during real-time simulation and single-sample inference. Checksums are computed using SHA-256 and verified upon loading.
