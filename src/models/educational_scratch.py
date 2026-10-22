"""
Educational Machine Learning Algorithms Implemented from Scratch using NumPy
Provides transparent, mathematically rigorous implementations of:
1. EducationalLogisticRegression (Gradient Descent, Binary Cross-Entropy, Sigmoid)
2. EducationalDecisionStump (Gini Impurity, Optimal Split Search, Gini Gain)

Intended for college robotics workshops to explain ML fundamentals.
"""
from typing import Any
import numpy as np


class EducationalLogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.

    Mathematical Formulation:
    - Hypothesis: z = X @ w + b
    - Sigmoid Activation: sigma(z) = 1 / (1 + exp(-z))
    - Loss: Binary Cross-Entropy (Log-Loss):
        J(w, b) = - (1/N) * sum[ y * log(p + eps) + (1 - y) * log(1 - p + eps) ]
    - Gradients:
        dJ/dw = (1/N) * X^T @ (p - y)
        dJ/db = (1/N) * sum(p - y)
    - Parameter Updates:
        w = w - alpha * dJ/dw
        b = b - alpha * dJ/db
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_iterations: int = 500,
        threshold: float = 0.5,
        eps: float = 1e-15,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.eps = eps
        self.random_state = random_state

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function to prevent exp() overflow.
        For z >= 0: 1 / (1 + exp(-z))
        For z < 0:  exp(z) / (1 + exp(z))
        """
        z_clipped = np.clip(z, -500.0, 500.0)
        return np.where(
            z_clipped >= 0,
            1.0 / (1.0 + np.exp(-z_clipped)),
            np.exp(z_clipped) / (1.0 + np.exp(z_clipped)),
        )

    def _compute_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Computes numerically stabilized Binary Cross-Entropy loss."""
        p = np.clip(y_prob, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EducationalLogisticRegression":
        """
        Fits weights and bias via batch gradient descent.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array of shape (n_samples, n_features), got shape {X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X and y length mismatch: {len(X_arr)} vs {len(y_arr)}")

        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        # Initialize small random weights
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            # 1. Forward pass
            linear_model = np.dot(X_arr, self.weights) + self.bias
            y_prob = self._sigmoid(linear_model)

            # 2. Compute loss
            loss = self._compute_loss(y_arr, y_prob)
            self.loss_history.append(loss)

            # 3. Gradient computation
            error = y_prob - y_arr
            dw = (1.0 / n_samples) * np.dot(X_arr.T, error)
            db = (1.0 / n_samples) * np.sum(error)

            # 4. Gradient descent update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability for class 1."""
        if self.weights is None:
            raise ValueError("Model must be fitted before predict_proba.")
        X_arr = np.asarray(X, dtype=np.float64)
        linear = np.dot(X_arr, self.weights) + self.bias
        p1 = self._sigmoid(linear)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, custom_threshold: float | None = None) -> np.ndarray:
        """Classifies observations based on decision threshold."""
        thresh = custom_threshold if custom_threshold is not None else self.threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)


class EducationalDecisionStump:
    """
    Educational 1-Split Decision Tree (Decision Stump) based on Gini Impurity.

    Mathematical Formulation:
    - Gini Impurity: G(S) = 1 - sum(p_k^2)
    - For binary classification: G(S) = 1 - (p_0^2 + p_1^2)
    - Split Quality (Weighted Child Impurity):
        G_split = (N_left / N) * G(S_left) + (N_right / N) * G(S_right)
    - Gini Gain (Reduction):
        Delta_G = G(Parent) - G_split
    """

    def __init__(self):
        self.best_feature_idx: int | None = None
        self.best_threshold: float | None = None
        self.left_prediction: int = 0
        self.right_prediction: int = 1
        self.gini_gain: float = 0.0
        self.parent_gini: float = 0.0

    @staticmethod
    def compute_gini(y: np.ndarray) -> float:
        """Calculates Gini impurity for a 1D binary label array."""
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return float(1.0 - (p0**2 + p1**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EducationalDecisionStump":
        """
        Exhaustively searches for the single feature and threshold that maximizes Gini Gain.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)

        n_samples, n_features = X_arr.shape
        self.parent_gini = self.compute_gini(y_arr)

        best_gain = -1.0

        for feat_idx in range(n_features):
            feat_vals = np.unique(X_arr[:, feat_idx])
            if len(feat_vals) <= 1:
                continue

            # Candidate thresholds as midpoints between sorted unique values
            sorted_vals = np.sort(feat_vals)
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            # Subsample candidate thresholds if too many for fast demonstration
            if len(thresholds) > 40:
                indices = np.linspace(0, len(thresholds) - 1, 40, dtype=int)
                thresholds = thresholds[indices]

            for thresh in thresholds:
                left_mask = X_arr[:, feat_idx] <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                gini_left = self.compute_gini(y_arr[left_mask])
                gini_right = self.compute_gini(y_arr[right_mask])

                weighted_gini = (np.sum(left_mask) / n_samples) * gini_left + (
                    np.sum(right_mask) / n_samples
                ) * gini_right

                gain = self.parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    self.best_feature_idx = feat_idx
                    self.best_threshold = float(thresh)
                    self.gini_gain = float(gain)
                    # Majority vote in each leaf
                    self.left_prediction = int(np.mean(y_arr[left_mask]) >= 0.5)
                    self.right_prediction = int(np.mean(y_arr[right_mask]) >= 0.5)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class using the learned decision stump threshold."""
        if self.best_feature_idx is None or self.best_threshold is None:
            raise ValueError("Stump must be fitted before predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        is_left = X_arr[:, self.best_feature_idx] <= self.best_threshold
        return np.where(is_left, self.left_prediction, self.right_prediction)
