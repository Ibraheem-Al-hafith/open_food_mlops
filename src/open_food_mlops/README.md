# 🧠 Open Food MLOps — Core Library

> The reusable library powering every pipeline, service, and test in the platform.

## 🎯 Purpose

This package contains **all business logic**. Pipelines orchestrate it; serving exposes it; tests verify it. Nothing here knows about HTTP, CLI, or Docker.

## 🧩 Structure

| Subpackage | Responsibility |
|------------|----------------|
| `config/` | Settings singleton, Pydantic schemas, feature schema |
| `data/` | Ingestion, chunked processing, splitting |
| `evaluation/` | Metric computation |
| `experiments/` | Orchestrator + champion selection engine |
| `features/` | Composable DataFrame transformers |
| `models/` | Base classes, registry, implementations, tuning |
| `tracking/` | MLflow wrapper |
| `utils/` | Logging |

## 🔌 Contract

- Every public module **must** be importable without side effects (no downloads, no training at import time).
- Settings come exclusively from `config.settings.settings`.
- Logging is configured via `setup_logging()` — never `print()`.

## 🧪 Usage

```python
from open_food_mlops.config.settings import settings
from open_food_mlops.experiments.orchestrator import ExperimentOrchestrator
from open_food_mlops.config.schemas import ExperimentPlan

plan = ExperimentPlan(**yaml.safe_load(open("config/experiment.yaml")))
result = ExperimentOrchestrator(plan).run()
print(result.champion.model_name)
```

## ⚙️ Design Decisions

- **Layered architecture** — data → features → models → experiments. Each layer imports only from below.
- **Registry pattern for models** — decorator-based discovery, no `if/elif` chains.
- **Pydantic everywhere at boundaries** — schemas guard configs and API contracts.
- **No side effects on import** — enables fast tests and lazy resource loading.

## 🚧 What Does NOT Belong Here

- HTTP routes → `serving/`
- CLI entrypoints → `main.py`, `pipelines/`
- Container configs → `docker/`

## 🔗 Related

- [Models subsystem](./models/README.md)
- [Features subsystem](./features/README.md)
- [Data subsystem](./data/README.md)