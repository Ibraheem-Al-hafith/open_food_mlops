# 🧠 Models

> Base abstractions, registry, concrete estimators, and tuning.

## 🎯 Purpose

Provide a unified `BaseModel` interface so the training pipeline never
touches sklearn/xgboost/lightgbm directly. Adding a model is one file
plus one decorator.

## 🧩 Structure

```
models/
├── base.py              # BaseModel ABC — the contract
├── registry.py          # @register decorator + get_model_class()
├── factory.py           # create_model() from config
├── specs.py             # SearchSpace primitives
├── wrapper.py           # NovaPipelineWrapper — MLflow PyFunc bundling
├── config.py            # ModelConfig, TuningConfig (Pydantic)
├── implementations/     # Concrete models
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── logistic_regression.py
│   ├── xgboost.py
│   └── lightgbm.py
└── tuning/
    ├── base.py          # BaseTuner ABC + TuningResult
    └── optuna_tuner.py  # Optuna TPE implementation
```

## 🔌 Model Contract (summary)

Every model **MUST**:
1. Inherit from `BaseModel`
2. Be decorated with `@register("<name>")`
3. Implement `fit`, `predict`, `get_search_space`, `_save`, `_load`
4. Return `pd.Series` from `predict`, indexed by `X.index`

> 📘 **Full contract, copy-paste template, and golden rules:**
> [ADDING_A_MODEL.md](./ADDING_A_MODEL.md)

## 🧪 Minimal Usage

```python
from open_food_mlops.models.factory import create_model
from open_food_mlops.models.config import ModelConfig

model = create_model(ModelConfig(name="random_forest", params={"n_estimators": 200}))
model.fit(X_train, y_train)
```

## 🚧 What Does NOT Belong Here

- Data loading → `data/`
- Feature engineering → `features/`
- Orchestration → `experiments/`

## 🔗 Related

- 📘 [Adding a New Model](./ADDING_A_MODEL.md)
- [Experiments subsystem](../experiments/README.md)
- [Features subsystem](../features/README.md)

