"""Built-in model implementations module.

Importing this package registers all built-in models with the central ModelRegistry.
"""

from .decision_tree import DecisionTreeModel
from .lightgbm import LightGBMModel
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel

__all__ = [
    "DecisionTreeModel",
    "LightGBMModel",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
]