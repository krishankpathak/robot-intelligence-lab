"""
Decision Tree Estimator for ROBOT INTELLIGENCE LAB
"""
from sklearn.tree import DecisionTreeClassifier


def create_decision_tree_model(
    max_depth: int | None = 6,
    min_samples_leaf: int = 10,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Creates a regularized DecisionTreeClassifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        criterion="gini",
    )
