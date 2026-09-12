# 📊 Evaluation

> Metric computation engine.

## 🎯 Purpose

Compute classification metrics in one place so training, tuning, and monitoring all report the same numbers.

## 🧩 Structure

```
evaluation/
├── evaluator.py   # Evaluator + EvaluationResult dataclass
└── README.md
```

## 🔌 Contract

`Evaluator.evaluate(y_true, y_pred)` returns an `EvaluationResult` with a `.metrics` dict containing:

- `accuracy`
- `precision` (macro)
- `recall` (macro)
- `macro_f1`

The `.primary_score` property reads the configured primary metric (default `macro_f1`).

## 🧪 Usage

```python
from open_food_mlops.evaluation.evaluator import Evaluator

evaluator = Evaluator(primary_metric="macro_f1")
result = evaluator.evaluate(y_true, y_pred)
print(result.metrics)
print(result.primary_score)
```

## ⚙️ Design Decisions

- **Macro averaging** — NOVA classes are imbalanced; macro prevents majority-class dominance.
- **`zero_division=0`** — silent failures are worse than zero scores.
- **`frozen` dataclass** — metrics are immutable once computed.

## 🚧 What Does NOT Belong Here

- Model selection logic → `experiments/selection.py`
- Drift detection → `monitoring/`
- Hyperparameter optimization → `models/tuning/`

## 🔗 Related

- [Selection engine](../experiments/selection.py)
- [Orchestrator](../experiments/orchestrator.py)