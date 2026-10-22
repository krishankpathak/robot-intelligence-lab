# 15–20 Minute College Robotics Workshop Script
## "Predictive Maintenance & Fault Analytics for Articulated Robots"

### Audience:
Undergraduate robotics and engineering students with foundational programming knowledge.

---

## ⏱️ Timeline Overview
- **Minute 0–3**: Introduction to Robot Predictive Maintenance & Local AI Policy
- **Minute 3–7**: The Danger of Data Leakage & The Class Imbalance Trap
- **Minute 7–12**: Live Sensor Lab Simulation (Injecting Faults & Diagnostics)
- **Minute 12–16**: Demystifying the Math (From-Scratch Sigmoid & Gini Impurity)
- **Minute 16–20**: Q&A and Engineering Trade-Offs

---

## Step-by-Step Workshop Delivery

### Minute 0–3: Setting the Stage
1. **What to Open**: Launch `streamlit run app.py` and open the **1. Overview & Architecture** tab.
2. **What to Say**:
   > *"Welcome everyone! Today we're looking at predictive maintenance on a 6-axis industrial robot arm. In an automated factory, when a joint motor burns out or jams, production stops. We want AI that monitors encoder position, motor temperature, and driver voltage to detect faults before catastrophic failure.*
   > *Notice the badge: '100% Local Inference • Zero Cloud APIs'. We aren't sending factory telemetry to external cloud APIs. Everything runs right here on local hardware."*

---

### Minute 3–7: Why Accuracy Lies & Preventing Data Leakage
1. **What to Open**: Navigate to **3. Exploratory Sensor Analysis** and then **4. Model Training Lab**.
2. **What to Click**: Show the Target Distribution Donut Chart (8.2% Faults vs 91.8% Normal).
3. **What to Explain**:
   > *"Look at this donut chart. Normal operation is over 90% of the data. If I build a 'dumb' model that always shouts 'NORMAL', it gets over 90% accuracy! But it catches zero faults. That's why in predictive maintenance, accuracy is useless. We care about **Recall** (catching every failure) and **F1-Score**.*
   > *Now look at the Data Partition Strategy. Why can't we just use random train/test split? Because our sensors sample at 10 Hz. If sample 100 goes to training and sample 101 goes to testing, they are 100 milliseconds apart. The model merely cheats by memorizing the neighbor. We use **Grouped Run Splitting** to hold out complete test runs."*

---

### Minute 7–12: Live Sensor Lab Simulation
1. **What to Open**: Click on **6. Live Sensor Lab (Simulation)**.
2. **What to Click**:
   - Select **Mode B: Synthetic Fault Scenarios**.
   - Choose scenario: **2. High Friction & Overheating Fault**.
   - Click **⏩ Stream +5 Steps** 2 or 3 times.
3. **What to Observe**:
   - Watch the temperature climb on the real-time line chart.
   - Watch the AI diagnosis card switch from **NORMAL (Green)** to **⚠️ FAULT DETECTED (Red)**.
   - Point out the local explanation box showing which sensor triggered the alert.
4. **What to Explain**:
   > *"Notice that the model didn't cheat. The simulation generated thermal drift, passed only the physical sensor readings into our saved scikit-learn pipeline, and the model independently detected that temperature crossed normal baseline thresholds."*

---

### Minute 12–16: Demystifying the Math
1. **What to Open**: Click on **8. Educational From-Scratch Lab**.
2. **What to Click**:
   - Go to Tab 1 (Logistic Regression): Adjust the learning rate slider and show the loss curve converging.
   - Go to Tab 2 (Decision Stump): Adjust the temperature candidate threshold slider and watch Gini Gain update in real time.
3. **What to Explain**:
   > *"AI isn't magic. In Tab 2, Gini impurity measures node disorder. When we split our dataset at 40°C, the impurity drops, yielding positive Gini Gain. A Decision Tree is simply an automated greedy search that finds the split giving the maximum Gini Gain."*

---

### Minute 16–20: Typical Student Questions & Accurate Answers

**Q1: "Can we use an LLM or ChatGPT to predict the motor faults instead of these models?"**  
*Answer*: No. LLMs process natural language text tokens. They are slow, non-deterministic, and cannot process 10 Hz continuous sensor streams in real-time edge microcontrollers. Classical ML models (Logistic Regression, Decision Trees) execute in microseconds with strict mathematical determinism.

**Q2: "Why didn't we use Deep Learning / Neural Networks with 50 layers?"**  
*Answer*: Occam's razor. For 3 tabular physical sensor channels, tree ensembles and linear classifiers achieve strong performance with instant training times, full interpretability, and low compute footprints. Deep networks risk overfitting on limited run sequences.

**Q3: "What happens if a sensor cable disconnects?"**  
*Answer*: That's an Out-of-Distribution (OOD) event. In Tab 7 (Predict a Sample), if an extreme glitch like -32,000 steps occurs, the OOD detector flags that the input is outside the physical training domain, alerting technicians to check the wiring.
