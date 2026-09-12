# ⚙️ Config

> Declarative YAML configuration and runtime settings for the Open Food MLOps platform.

## 🎯 Purpose

This folder holds **all configuration that drives the platform**. Nothing here is code — it's the source of truth that the orchestrator, logger, and pipelines read at runtime.

## 🧩 Structure

```
config/
├── experiment.yaml   # Experiment plan: models, tuning, gates, splits
├── logging.yaml      # Python logging config (handlers, formatters, rotation)
└── README.md
```

## 🔌 Contract

- `experiment.yaml` is validated by `open_food_mlops.config.schemas.ExperimentPlan`
- `logging.yaml` is consumed by `open_food_mlops.utils.logger.setup_logging()`
- **Any key added to `experiment.yaml` must have a matching Pydantic field**

## 🧪 Usage

```bash
python main.py train --config config/experiment.yaml
```

Adding a new model to the experiment:

```yaml
models:
  - name: my_new_model
    enabled: true
    params: {}
    tuning:
      enabled: true
      method: optuna
      trials: 20
```

## ⚙️ Design Decisions

- **YAML over Python configs** — non-engineers can tune experiments.
- **`extra="forbid"` in Pydantic** — typos fail loudly, not silently.
- **Logging rotation by midnight, 30 backups** — prevents disk exhaustion in production.

## 🚧 What Does NOT Belong Here

- Secrets (use `.env` + `settings.py`)
- Runtime state (use `data/`)
- Model hyperparameters (those live in `experiment.yaml` under `models[].params`)

## 🔗 Related

- [Settings singleton](../src/open_food_mlops/config/settings.py)
- [Pydantic schemas](../src/open_food_mlops/config/schemas.py)
