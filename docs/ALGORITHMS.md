# Machine Learning Algorithms & Mathematical Foundations

This guide explains the core algorithms implemented in **ROBOT INTELLIGENCE LAB**, distinguishing educational mathematical principles from production-grade library optimizations.

---

## 1. Logistic Regression
A foundational linear classifier that models the probability that an observation belongs to the positive fault class ($y=1$).

### Mathematical Principle:
1. **Linear Combination**:
   $$z = \mathbf{w}^T \mathbf{x} + b = w_1 x_{\text{pos}} + w_2 x_{\text{temp}} + w_3 x_{\text{volt}} + b$$
2. **Sigmoid Activation**: Compresses the real line into a probability interval $(0, 1)$:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
3. **Binary Cross-Entropy Loss (Log-Loss)**:
   $$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln(p^{(i)} + \epsilon) + (1 - y^{(i)}) \ln(1 - p^{(i)} + \epsilon) \right]$$
4. **Gradient Descent**: Updates parameters along the steepest descent direction:
   $$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\mathbf{p} - \mathbf{y}), \quad \mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

### Educational vs. Production Implementation:
- **Educational (`EducationalLogisticRegression`)**: Uses standard first-order batch gradient descent with fixed learning rate $\alpha$. Ideal for students to observe loss decreasing step-by-step.
- **Production (`sklearn.linear_model.LogisticRegression`)**: Employs quasi-Newton methods like L-BFGS, which approximate the inverse Hessian matrix to accelerate convergence with adaptive step sizes and L2 regularization.

---

## 2. Decision Tree & Gini Impurity
Decision trees recursively partition feature space using orthogonal split boundaries.

### Mathematical Principle (Gini Impurity):
Measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution:
$$G(S) = 1 - \sum_{k=1}^K p_k^2 = 1 - (p_0^2 + p_1^2)$$
- A pure node containing only normal samples has $G(S) = 1 - (1^2 + 0^2) = 0.0$.
- A completely mixed 50/50 node has $G(S) = 1 - (0.5^2 + 0.5^2) = 0.5$.

### Gini Reduction (Gain):
A split evaluates candidate thresholds $t$ across features to maximize impurity reduction:
$$\Delta G = G(\text{Parent}) - \left[ \frac{N_{\text{left}}}{N} G(\text{Left}) + \frac{N_{\text{right}}}{N} G(\text{Right}) \right]$$

### Educational Stump vs. Deep Trees:
- **Educational (`EducationalDecisionStump`)**: Evaluates a single optimal threshold across all sensors, teaching students how impurity reduction selects the most discriminative sensor.
- **Production (`sklearn.tree.DecisionTreeClassifier`)**: Constructs a multi-level binary tree with recursive greedy partitioning, cost-complexity pruning, and leaf sample constraints.

---

## 3. Random Forest
An ensemble of decorrelated decision trees built using **Bootstrap Aggregating (Bagging)** and **Random Subspace Sampling**.
1. **Bootstrap Samples**: Each tree is trained on an independent random sample drawn with replacement from the training set.
2. **Feature Randomness**: At each candidate split, only a random subset of sensors (typically $\sqrt{D}$) is considered. This prevents one dominant sensor (e.g. voltage drop) from overshadowing subtle thermal interactions.
3. **Voting**: Predictions are aggregated via majority vote or average class probability.

---

## 4. Gradient Boosting
Builds trees sequentially rather than independently:
- Tree 1 makes initial predictions.
- Tree 2 is trained to predict the **pseudo-residuals** (negative gradients of the loss function) of Tree 1.
- Each subsequent tree corrects the remaining errors of the ensemble.
- `HistGradientBoostingClassifier` bins continuous sensor values into 256 discrete integer bins, accelerating training on hundreds of thousands of samples by orders of magnitude.

---

## 5. Classification Metrics Under Class Imbalance

| Metric | Formula | Practical Meaning in Robotics |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Misleading when faults are rare. A broken robot predicted normal still counts toward high accuracy. |
| **Precision** | $\frac{TP}{TP + FP}$ | Of all alarms triggered, how many were real faults? High precision minimizes false maintenance downtime. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Of all actual robot breakdowns, how many were caught? Critical for safety and preventing hardware destruction. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall. Best single metric for imbalanced fault detection. |
| **PR-AUC** | Area under Precision-Recall Curve | Summarizes trade-offs across all possible decision thresholds; unaffected by large counts of true negatives. |
| **ROC-AUC** | Area under True Positive vs False Positive Rate | Evaluates global class separability. |
