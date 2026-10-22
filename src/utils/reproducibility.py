"""
Reproducibility Utilities for ROBOT INTELLIGENCE LAB
Ensures deterministic behavior across data splitting, models, and NumPy.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across python runtime, os, and numpy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int = 42) -> np.random.Generator:
    """Returns a modern numpy random Generator instance."""
    return np.random.default_rng(seed)
