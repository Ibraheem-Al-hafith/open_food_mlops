# 📈 Tracking

> MLflow wrapper for tracking, registry, and alias governance.

## 🎯 Purpose

Encapsulate every MLflow call behind a small façade so the orchestrator never talks to MLflow directly. This makes swapping tracking backends (or mocking in tests) trivial.

## 🧩 Structure

```
tracking/
├── mlflow_tracker.py   # MLflowTracker façade
└── README.md
```

## 🔌 Contract

`MLflowTracker` exposes:

- `start_run(run_name)` — context manager for a run
- `log_params(dict)` / `log_metrics(dict)` / `log_artifact(path)`
- `register_and_alias_pyfunc(...)` — logs a PyFunc, registers it, sets an alias

Registered model name is **fixed**: `open_food_champion` with alias `production`.

## 🧪 Usage

```python
from open_food_mlops.tracking.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(
    tracking_uri=settings.mlflow_tracking_uri,
    experiment_name="my-experiment",
)

with tracker.start_run(run_name="baseline"):
    tracker.log_params({"max_depth": 10})
    tracker.log_metrics({"macro_f1": 0.72})
```

## ⚙️ Design Decisions

- **Alias over stage** — MLflow stages are deprecated; aliases are the modern primitive.
- **PyFunc bundling** — the feature pipeline travels with the model.
- **Façade pattern** — enables unit testing without an MLflow server.

## 🚧 What Does NOT Belong Here

- Model training → `experiments/`
- Experiment selection logic → `experiments/selection.py`

## 🔗 Related

- [Orchestrator](../experiments/orchestrator.py)
- [Model wrapper](../models/wrapper.py)