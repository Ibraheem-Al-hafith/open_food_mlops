# 🎛️ Adding a New Tuner

This guide explains how to create a custom Hyperparameter Optimization (HPO) engine. Whether you want to use Optuna, Ray Tune, Grid Search, or a custom Bayesian Optimization library, you can plug it into the pipeline by implementing the `BaseTuner` contract.

## 📋 The 2-Step Process

1. **Create** your tuner class in the `tuning/` directory.
2. **Implement** the `optimize` method to return a standardized `TuningResult`.

---

## Step 1 & 2: Create Your Tuner

Create a new file (e.g., `tuning/my_custom_tuner.py`). Your tuner must inherit from `BaseTuner` and implement the `optimize` method, which receives an `objective` function. 

The `objective` function takes a dictionary of parameters and returns a float (the validation score). Your job is to sample parameters, call the objective, and track the best result.

### 🟢 Copy-Paste Tuner Template

```python
"""Custom Hyperparameter Optimization Engine."""
from __future__ import annotations

import logging
from typing import Any, Callable

from .base import BaseTuner, TuningResult
from ..specs import SearchSpace

logger = logging.getLogger(__name__)

class MyCustomTuner(BaseTuner):
    """A custom tuner implementation (e.g., Random Search, Grid Search, etc.)."""
    
    def __init__(
        self,
        model_class: type,
        search_space: SearchSpace,
        n_trials: int,
        my_custom_argument: str = "default_value",
    ) -> None:
        # 👇 1. ALWAYS call super().__init__ to store base attributes
        super().__init__(model_class, search_space, n_trials)
        self.my_custom_argument = my_custom_argument

    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
    ) -> TuningResult:
        """
        Run the hyperparameter optimization loop.
        
        Args:
            objective: A callable that takes a dict of parameters and 
                       returns a float score (higher is better).
                       
        Returns:
            TuningResult containing the best parameters and score.
        """
        best_params: dict[str, Any] = {}
        best_score = float("-inf")
        
        # 👇 2. Implement your sampling and evaluation logic here
        for trial in range(self.n_trials):
            # Example: Sample parameters based on self.search_space
            # params = self._sample_parameters(self.search_space)
            
            # Example: Evaluate the pipeline using the objective function
            # score = objective(params)
            
            # Example: Track the best result
            # if score > best_score:
            #     best_score = score
            #     best_params = params
            
            logger.info(f"Trial {trial + 1}/{self.n_trials} completed.")
            
        # 👇 3. Return the standardized TuningResult dataclass
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=self.n_trials,
        )
        
    def _sample_parameters(self, space: SearchSpace) -> dict[str, Any]:
        """Helper method to sample parameters based on the SearchSpace specs."""
        params = {}
        for name, spec in space.items():
            if spec.type == "int":
                params[name] = random.randint(spec.low, spec.high)
            elif spec.type == "float":
                params[name] = random.uniform(spec.low, spec.high)
            elif spec.type == "categorical":
                params[name] = random.choice(spec.choices)
        return params
```

---

## How to Use Your Tuner

Because tuners are instantiated directly by the pipeline orchestrator rather than a global registry, you simply pass your custom tuner class into the pipeline configuration or orchestrator initialization.

```python
# Example usage in your pipeline orchestrator
from tuning.my_custom_tuner import MyCustomTuner
from models.implementations.lightgbm import LightGBMModel

# 1. Define the search space (usually fetched via model_class.get_search_space())
search_space = LightGBMModel.get_search_space()

# 2. Instantiate your custom tuner
tuner = MyCustomTuner(
    model_class=LightGBMModel,
    search_space=search_space,
    n_trials=50,
    my_custom_argument="fast_mode"
)

# 3. Define the objective function (usually handled by the orchestrator)
def objective(params: dict) -> float:
    model = LightGBMModel(params)
    model.fit(X_train, y_train)
    return model.evaluate(X_val, y_val)

# 4. Run optimization
result = tuner.optimize(objective)
print(f"Best Params: {result.best_params}")
print(f"Best Score: {result.best_score}")
```

### 💡 Golden Rules for Tuners
1. **Respect the `SearchSpace`**: Use the `type` attribute (`"int"`, `"float"`, `"categorical"`) of the specs in `self.search_space` to know how to sample parameters.
2. **Return `TuningResult`**: Always return the frozen `TuningResult` dataclass. The orchestrator expects this exact structure to log results to MLflow.
3. **Maximization Assumption**: The base pipeline assumes higher scores are better (`direction="maximize"` in `TuningConfig`). If your underlying library minimizes by default (like some loss functions), ensure you negate the score before returning it to the `objective` or handle it inside your tuner.
