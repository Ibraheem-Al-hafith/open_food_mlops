# 🤖 Adding a New Model

This guide explains how to add a new machine learning model to the project. By following the `BaseModel` contract and using the `@register` decorator, your model will automatically integrate with the training pipeline, hyperparameter tuning engine, and MLflow tracking without changing any core orchestration code.

## 📋 The 3-Step Process

1. **Create** your model class in the `implementations/` directory.
2. **Decorate** it with `@register("your_model_name")`.
3. **Import** it in `implementations/__init__.py` to trigger registration at startup.

---

## Step 1 & 2: Create Your Model

Create a new file (e.g., `implementations/my_custom_model.py`) and use the template below. 

Your model **must** inherit from `BaseModel` and implement the core abstract methods: `fit`, `predict`, `get_search_space`, `_save`, and `_load`.

### 🟢 Copy-Paste Model Template

```python
"""My Custom Model implementation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)

# 👇 1. Register the model with a unique, lowercase string name
@register("my_custom_model")
class MyCustomModel(BaseModel):
    """Adapter for My Custom Model satisfying the BaseModel interface."""
    
    # 👇 2. Define the class variable matching the registry name
    model_name = "my_custom_model"

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default parameters for the underlying estimator."""
        return {
            "param_a": 10,
            "param_b": "auto",
            "random_state": 42,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the underlying estimator."""
        # Merge defaults with user-provided config
        params = {**self.get_default_params(), **self.config}
        
        # 👇 3. Initialize and fit your underlying library's model
        # self.estimator_ = MyUnderlyingEstimator(**params)
        # self.estimator_.fit(X, y)
        
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        self._check_is_fitted()
        
        # 👇 4. Return a pandas Series with the original index preserved
        # predictions = self.estimator_.predict(X)
        # return pd.Series(predictions, index=X.index, name="prediction")
        pass

    @classmethod
    def get_search_space(cls) -> SearchSpace:
        """Define the hyperparameter search space for the Tuning Engine."""
        return {
            "param_a": IntParameter(low=1, high=100),
            "param_b": CategoricalParameter(choices=("auto", "manual", "advanced")),
            "learning_rate": FloatParameter(low=0.001, high=0.1, log=True),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted model state to disk."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore the fitted model state from disk."""
        payload = joblib.load(path / "model.joblib")
        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True
        return model
```

---

## Step 3: Register It Globally

For the pipeline to "see" your new model, Python needs to execute the `@register` decorator. Open `implementations/__init__.py` and add your import:

```python
# implementations/__init__.py
from .decision_tree import DecisionTreeModel
from .lightgbm import LightGBMModel
# ... existing imports ...

# 👇 Add your new model import here!
from .my_custom_model import MyCustomModel

__all__ = [
    # ... existing models ...
    "MyCustomModel",
]
```

### 💡 Golden Rules for Models
1. **Always set `self.is_fitted_ = True`** at the end of your `fit()` method. The base class relies on this to prevent inference on unfitted models.
2. **Preserve Pandas Indexes**: When returning predictions in `predict()`, always pass `index=X.index` to the `pd.Series` constructor. This prevents alignment bugs downstream.
3. **Search Space Types**: Use the provided dataclasses (`IntParameter`, `FloatParameter`, `CategoricalParameter`) from `specs.py`. The tuning engine reads these natively.
