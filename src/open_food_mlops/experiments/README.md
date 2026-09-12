> 🔥 **Critical** — this is where end-to-end orchestration lives.

# 🧪 Experiments

> Orchestration, champion selection, and MLflow promotion.

## 🎯 Purpose

Wire together data loading, splitting, tuning, training, evaluation, and promotion into a single reproducible run.

## 🧩 Structure

```
experiments/
├── orchestrator.py   # ExperimentOrchestrator — end-to-end run
├── selection.py      # Quality gates + champion ranking
└── README.md
```

## 🔌 Contract

`ExperimentOrchestrator(plan).run()` returns a `SelectionResult`:

```python
SelectionResult(
    champion: CandidateResult | None,
    passed_candidates: list[CandidateResult],
    rejected_candidates: list[CandidateResult],
)
```

The orchestrator:
1. Loads and splits data
2. For each enabled model: tunes (optional) → cross-validates → refits on full train
3. Applies quality gates
4. Promotes the champion to `open_food_champion@production`

## 🧪 Usage

```python
from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.experiments.orchestrator import ExperimentOrchestrator
import yaml

plan = ExperimentPlan(**yaml.safe_load(open("config/experiment.yaml")))
result = ExperimentOrchestrator(plan).run()
print(result.champion.model_name, result.champion.metrics)
```

## ⚙️ Design Decisions

- **Single refit on full training data** — no duplicate computation after CV.
- **Quality gates before ranking** — a model that misses a gate can't be champion.
- **MLflow alias over stage** — `production` alias is the modern governance primitive.
- **`NovaPipelineWrapper` on promotion** — bundles feature pipeline + estimator into one PyFunc.
- **Per-model try/except** — one model failing doesn't kill the experiment.

## 🚧 What Does NOT Belong Here

- Model implementations → `models/implementations/`
- Feature transformations → `features/`
- HTTP or CLI glue → `pipelines/`, `main.py`

## 🔗 Related

- [Models subsystem](../models/README.md)
- [Features subsystem](../features/README.md)
- [MLflow tracker](../tracking/mlflow_tracker.py)
