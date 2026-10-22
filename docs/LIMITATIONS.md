# System Limitations & Engineering Constraints

## 1. Scope of Model Predictions

### What the Model Predicts:
- **Binary Health Status**: Estimates whether current sensor readings (`position`, `temperature`, `voltage`) exhibit anomalous patterns consistent with known training fault conditions.
- **Estimated Fault Probability**: Reflects statistical confidence under the fitted decision surface.
- **Primary Telemetry Drivers**: Quantifies which sensor reading deviates furthest from empirical training population baselines.

### What the Model CANNOT Predict:
- **Specific Fault Root Causes**: The dataset provides binary `0/1` failure labels, not multi-class labels for "bearing race pit", "shorted stator coil", "gearbox backlash", or "lubrication dryout". The classifier cannot distinguish specific physical breakdown modes without validated labeled data for each failure mode.
- **Remaining Useful Life (RUL)**: The model classifies instantaneous observations. It does not calculate time-to-failure curves or RUL degradation trajectories.
- **Guaranteed Physical Diagnostics**: A statistical prediction of `label = 1` indicates an abnormal telemetry signature, not guaranteed mechanical teardown proof. Technicians must conduct physical inspections.

---

## 2. Dataset Limitations
1. **Limited Operating Sequences**: The dataset comprises 23 training sequences and 8 test sequences. Although there are 235,854 timestamped samples, they originate from a small number of physical experimental sessions.
2. **Domain/Distribution Shift**: Training data was collected under specific robot motions ("pick and place", "turning motor 6", "not moving"). If the robot is deployed on a radically different motion trajectory or carried payload, sensor distributions will shift, potentially triggering false alarms.
3. **Severe Class Imbalance**: With only 7.2% positive fault samples, false positive rates must be managed carefully by tuning decision thresholds according to operational cost.

---

## 3. Simulation Constraints
- The interactive simulation is an educational telemetry generator, not a high-fidelity finite-element or multi-body dynamics simulator (such as Isaac Sim or Gazebo).
- Synthetic scenarios are parameterized approximations designed to illustrate model inference workflows. They do not simulate complex multi-physics fluid dynamics or thermal radiation transfer.

---

## 4. Deployment Prerequisites
Before deploying this model to a physical robot cell:
- Integrate sliding-window sequence features (as prototyped in `utility.py`).
- Implement sensor sanity hardware checks to reject physical wiring dropouts before model inference.
- Re-calibrate probability decision thresholds against operational economics (cost of false alarms vs cost of unpredicted stoppage).
