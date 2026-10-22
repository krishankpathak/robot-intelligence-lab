"""
Data Transformations and Preprocessing Pipelines for ROBOT INTELLIGENCE LAB
Builds scikit-learn compatible transformation pipelines fitted strictly on training data.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from config.settings import PRIMARY_SENSOR_FEATURES, MOTOR_COLUMN


def get_scaler(scaler_type: str = "standard"):
    """Returns a scikit-learn scaler instance."""
    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }
    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler type '{scaler_type}'. Choose from {list(scalers.keys())}")
    return scalers[scaler_type]


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that combines imputation, scaling, and encoding.
    Ensures that transformations can be saved and loaded alongside the model.
    """
    if numeric_features is None:
        numeric_features = PRIMARY_SENSOR_FEATURES

    transformers = []

    # Numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    scaler = get_scaler(scaler_type)
    if scaler != "passthrough":
        num_steps.append(("scaler", scaler))

    transformers.append(("num", Pipeline(steps=num_steps), numeric_features))

    # Categorical pipeline (e.g. motor id)
    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(steps=cat_steps), categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
