"""
Logistic Regression Estimator for ROBOT INTELLIGENCE LAB
"""
from sklearn.linear_model import LogisticRegression


def create_logistic_regression_model(
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Creates a configured scikit-learn LogisticRegression classifier."""
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
