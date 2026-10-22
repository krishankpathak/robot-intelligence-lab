# Troubleshooting & FAQ Guide

## 1. Missing Raw Kaggle Dataset
- **Symptom**: `FileNotFoundError: Training directory does not exist: D:\ai_analysis\Data\robot-predictive-maintenance`
- **Solution**: The application automatically falls back to the documented synthetic benchmark dataset in `data/sample/synthetic_robot_telemetry.csv`. If you have downloaded the Kaggle files, place them in `d:/ai_analysis/Data/robot-predictive-maintenance` or update `RAW_KAGGLE_DIR` in `config/settings.py`.

---

## 2. Port Conflict When Launching Streamlit
- **Symptom**: `Address already in use: :::8501`
- **Solution**: Run Streamlit on a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
  Or terminate existing processes using port 8501:
  ```powershell
  # Windows PowerShell
  Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
  Stop-Process -Id <PID> -Force
  ```

---

## 3. Model Artifact Integrity / Checksum Mismatch
- **Symptom**: `ValueError: Security/Integrity Error: Model artifact checksum mismatch! Artifact may be tampered.`
- **Cause**: The `.joblib` model file was modified, corrupted, or replaced without regenerating its metadata.
- **Solution**: Retrain the model cleanly to generate a fresh cryptographic SHA-256 hash:
  ```bash
  python scripts/train_verified_models.py
  ```

---

## 4. Running the Automated Test Suite
- To run all 34 automated unit and integration tests:
  ```bash
  pytest tests -v
  ```
- To run a specific test module:
  ```bash
  pytest tests/test_models.py -v
  pytest tests/test_educational.py -v
  pytest tests/test_simulation.py -v
  ```

---

## 5. Memory Management in Live Simulation
- If browser rendering becomes sluggish after hours of simulation, click **🔄 Reset Buffer** in the Live Sensor Lab tab.
- The simulation engine uses a bounded circular deque (`maxlen=150`) to guarantee that memory consumption never grows indefinitely.
