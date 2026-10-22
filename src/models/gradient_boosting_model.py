"""
Gradient Boosting Estimator for ROBOT INTELLIGENCE LAB
Uses HistGradientBoostingClassifier for ultra-fast training and native handling of class weights.
"""
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier


def create_gradient_boosting_model(
    model_variant: str = "hist",
    learning_rate: float = 0.1,
    max_iter: int = 100,
    class_weight: str | None = "balanced",
    random_state: int = 42,
):
    """Creates a Gradient Boosting classifier."""
    if model_variant == "hist":
        return HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
    return GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=max_iter,
        random_state=random_state,
    )
