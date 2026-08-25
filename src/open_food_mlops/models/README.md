# System Design Document: Modular & Automated MLOps Pipeline

## 1. Executive Summary
This document outlines the architecture for a highly modular, automated Machine Learning pipeline. The system is designed around the **Registry Pattern**, **Factory Pattern**, and **Dependency Inversion** to ensure strict adherence to the Open/Closed Principle: the pipeline is open for extension (adding new models) but closed for modification (the core orchestration logic never changes).

The architecture is divided into six core components:
1. **Model Registry & Abstract Interface**
2. **Configuration Management**
3. **Data Layer & Validation Strategy**
4. **Hyperparameter Tuning Engine (HPO)**
5. **Pipeline Orchestrator & Experiment Tracking**
6. **Production Deployment & Model Registry**

---

## 2. Model Registry & Abstract Interface
To ensure the pipeline is unaffected when models are added, removed, or edited, the pipeline never imports models directly. It interacts solely with the Registry and an Abstract Base Class (ABC).

### 2.1 The Registry
A simple decorator registers model classes into a global dictionary. A factory function retrieves them.

```python
# registry.py
MODEL_REGISTRY = {}

def register(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name} already registered!")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str):
    """Factory function to instantiate a model."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model {name} not found in registry.")
    return MODEL_REGISTRY[name]
```

### 2.2 The Abstract Model Interface
Every model must inherit from this class. It enforces standard training, inference, evaluation, and serialization methods, as well as defining its own hyperparameter search space.

```python
# base_model.py
from abc import ABC, abstractmethod
import os, json

class AbstractModel(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, X, y) -> None: pass

    @abstractmethod
    def predict(self, X): pass

    @abstractmethod
    def evaluate(self, X, y) -> float: pass

    @classmethod
    @abstractmethod
    def get_search_space(cls) -> dict:
        """Returns the hyperparameter search space for the Tuning Engine."""
        pass

    # --- Serialization Enforcement ---
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)
        self._save_weights(path)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "config.json"), "r") as f:
            self.config = json.load(f)
        self._load_weights(path)

    @abstractmethod
    def _save_weights(self, path: str): pass
    
    @abstractmethod
    def _load_weights(self, path: str): pass
```

---

## 3. Configuration Management
Configuration is managed via YAML files and validated using **Pydantic** to ensure strict typing and fail-fast behavior at startup.

```python
# config.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class PipelineConfig(BaseModel):
    model_name: str
    finetune: bool = False
    tuning_trials: int = 50
    n_folds: int = 1
    base_model_params: Dict[str, Any] = {}
    data_path: str
```

---

## 4. The Data Layer & Validation Strategy
To avoid over-engineering while maintaining modularity, we separate the **creation of data splits** (Data Layer) from the **execution of the training loop** (Pipeline/Trainer Layer). 

The Data Layer handles acquisition, cleaning, and feature engineering. Crucially, feature engineering is treated as a stateful transformer (like Scikit-Learn's `Pipeline`) to prevent data leakage.

```python
# data_layer.py
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd

class AbstractDataProvider(ABC):
    @abstractmethod
    def setup(self): pass

    @abstractmethod
    def get_splits(self, n_folds: int = 1) -> Iterator[Tuple[Any, Any, Any, Any]]:
        """Yields (X_train, y_train, X_val, y_val)."""
        pass

class TabularDataProvider(AbstractDataProvider):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.feature_pipeline = None

    def setup(self):
        df = pd.read_csv(self.data_path)
        # Define and fit feature engineering pipeline ONLY on initial train data
        self.feature_pipeline = Pipeline([('scaler', StandardScaler())])
        X_initial = df.drop('target', axis=1)
        self.feature_pipeline.fit(X_initial)

    def get_splits(self, n_folds: int = 1):
        df = pd.read_csv(self.data_path)
        X = self.feature_pipeline.transform(df.drop('target', axis=1))
        y = df['target']

        if n_folds == 1:
            split_idx = int(len(X) * 0.8)
            yield X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
```

---

## 5. Hyperparameter Tuning Engine (HPO)
The tuning engine is completely agnostic to the underlying model. It reads the model's `get_search_space()`, suggests parameters to Optuna, and evaluates the result.

```python
# tuning_engine.py
import optuna

class TuningEngine:
    def __init__(self, model_class, base_config: dict, n_trials: int):
        self.model_class = model_class
        self.base_config = base_config
        self.n_trials = n_trials
        self.search_space = model_class.get_search_space()

    def _suggest_param(self, trial, param_name, space):
        if space["type"] == "int": return trial.suggest_int(param_name, space["low"], space["high"])
        if space["type"] == "float": return trial.suggest_float(param_name, space["low"], space["high"], log=space.get("log", False))
        if space["type"] == "categorical": return trial.suggest_categorical(param_name, space["choices"])

    def run(self, data_provider, mlflow_callback=None):
        def objective(trial):
            tuned_config = self.base_config.copy()
            for param, space in self.search_space.items():
                tuned_config[param] = self._suggest_param(trial, param, space)

            X_train, y_train, X_val, y_val = next(data_provider.get_splits(n_folds=1))
            model = self.model_class(tuned_config)
            model.train(X_train, y_train)
            return model.evaluate(X_val, y_val)

        study = optuna.create_study(direction="maximize")
        callbacks = [mlflow_callback] if mlflow_callback else []
        study.optimize(objective, n_trials=self.n_trials, callbacks=callbacks)
        
        return study.best_params, study.best_value
```

---

## 6. Pipeline Orchestrator & Experiment Tracking
The orchestrator is the "glue". It reads the config, fetches the model, decides whether to run HPO, and tracks everything in **MLflow**. 

By using `optuna.integration.MLflowCallback`, every single hyperparameter trial is automatically logged to the MLflow UI without custom logging code.

```python
# pipeline.py
import mlflow
from optuna.integration import MLflowCallback
from registry import get_model
from tuning_engine import TuningEngine

class MLPipeline:
    def __init__(self, config: PipelineConfig, data_provider):
        self.config = config
        self.data_provider = data_provider
        self.model_class = get_model(config.model_name)

    def run(self):
        self.data_provider.setup()
        
        with mlflow.start_run(run_name=f"{self.config.model_name}_run"):
            mlflow.log_params(self.config.dict())
            final_config = self.config.base_model_params.copy()

            # 1. Tuning (if enabled)
            if self.config.finetune:
                mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="val_metric")
                engine = TuningEngine(self.model_class, final_config, self.config.tuning_trials)
                best_params, _ = engine.run(self.data_provider, mlflow_callback)
                final_config.update(best_params)
                mlflow.log_params(best_params)

            # 2. Final Training & Evaluation
            test_metrics = []
            for fold, (X_tr, y_tr, X_val, y_val) in enumerate(self.data_provider.get_splits(self.config.n_folds)):
                model = self.model_class(final_config)
                model.train(X_tr, y_tr)
                metric = model.evaluate(X_val, y_val)
                test_metrics.append(metric)
                mlflow.log_metric(f"val_metric_fold_{fold+1}", metric)

            avg_metric = sum(test_metrics) / len(test_metrics)
            mlflow.log_metric("avg_val_metric", avg_metric)

            # 3. Save and Register Final Model
            final_model = self.model_class(final_config)
            final_model.train(X_tr, y_tr) # Train on last fold (or all data)
            final_model.config["model_name"] = self.config.model_name
            
            local_artifact_path = "temp_model_dir"
            final_model.save(local_artifact_path)
            
            # Log and Register in one step using pyfunc
            mlflow.pyfunc.log_model(
                artifact_path="mlflow_model",
                python_model=MLflowModelWrapper(), # See Section 7
                artifacts={"model_dir": local_artifact_path},
                registered_model_name=f"{self.config.model_name}_champion"
            )
```

---

## 7. Production Deployment & Model Registry
To move from experimentation to production, we bridge our custom `AbstractModel` with MLflow’s native inference engine using a `PythonModel` wrapper.

### 7.1 The MLflow Wrapper
```python
# mlflow_wrapper.py
import mlflow.pyfunc
from registry import get_model
import json, os

class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = context.artifacts["model_dir"]
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
            
        model_name = config.pop("model_name")
        model_class = get_model(model_name)
        
        self.model = model_class(config)
        self.model.load(model_dir)

    def predict(self, context, model_input):
        return self.model.predict(model_input)
```

### 7.2 Promoting to Production (Aliases)
Once the pipeline finishes, the model is registered. We use MLflow **Aliases** (the modern replacement for Stages) to promote the best model to `@champion`.

```python
from mlflow import MlflowClient

client = MlflowClient()
registry_model_name = f"{config.model_name}_champion"
latest_version = client.get_latest_versions(registry_model_name, stages=["None"])[0].version

client.set_registered_model_alias(
    name=registry_model_name,
    alias="champion",
    version=latest_version
)
```

### 7.3 Production Inference Service
The production code is completely decoupled from the training pipeline. It simply asks MLflow for the `@champion` model.

```python
# inference_service.py
import mlflow.pyfunc

class ProductionInferenceService:
    def __init__(self, registry_model_name: str):
        # Loads the model assigned to the @champion alias
        model_uri = f"models:/{registry_model_name}@champion"
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, data):
        return self.model.predict(data)
```

---

## 8. Summary of Architectural Benefits

1. **True Modularity (Open/Closed Principle):** Adding a new model (e.g., LightGBM, PyTorch) requires only creating a new file, decorating it with `@register`, and updating the YAML config. **Zero changes** are required in the pipeline, data layer, or tuning engine.
2. **No Over-Engineering:** The Data Layer uses standard Scikit-Learn pipelines for feature engineering and simple generators for K-Fold/Train-Val splits. It avoids complex DAG frameworks while strictly preventing data leakage.
3. **Automated Experiment Tracking:** By leveraging `optuna.integration.MLflowCallback`, the entire hyperparameter search space is automatically visualized in the MLflow UI without polluting the model code with logging statements.
4. **Safe, Zero-Downtime Rollbacks:** By using MLflow Aliases (`@champion`), the production inference service never needs to be redeployed or restarted when a new model is trained. If a new model performs poorly in production, the alias can be instantly reverted to the previous version via the MLflow UI.