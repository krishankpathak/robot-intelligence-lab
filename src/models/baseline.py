"""
Baseline Classifier Models for ROBOT INTELLIGENCE LAB
Establishes naive performance baselines (Majority Class, Constant, Stratified Random).
"""
from sklearn.dummy import DummyClassifier


def create_baseline_model(strategy: str = "most_frequent", random_state: int = 42) -> DummyClassifier:
    """
    Creates a naive baseline classifier.
    Strategies:
    - 'most_frequent': Always predicts the majority class (e.g. 0).
    - 'stratified': Generates predictions matching the training class distribution.
    - 'uniform': Generates random uniform 50/50 predictions.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)
