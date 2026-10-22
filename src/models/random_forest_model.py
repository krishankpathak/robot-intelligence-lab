"""
Random Forest Estimator for ROBOT INTELLIGENCE LAB
"""
from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model(
    n_estimators: int = 100,
    max_depth: int | None = 10,
    min_samples_leaf: int = 5,
    class_weight: str | None = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Creates a balanced Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
