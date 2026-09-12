# ⚙️ Core Config

> Runtime settings, Pydantic schemas, and the canonical feature schema.

## 🎯 Purpose

Every path, URL, and env-driven value used anywhere in the library flows through this package.

## 🧩 Structure

```
config/
├── settings.py    # Settings singleton (env-driven, path-resolving)
├── schemas.py     # ExperimentPlan, DataConfig, SelectionConfig, ...
├── features.py    # FEATURE_COLUMNS, TARGET_COLUMN — the schema contract
└── README.md
```

## 🔌 Contract

- **`settings`** is a module-level singleton — import it, never instantiate `Settings` yourself.
- **`FEATURE_COLUMNS`** is the *single source of truth* for input features. Adding a feature means editing this list **and** the ingestion pipeline.
- **`ExperimentPlan`** validates `config/experiment.yaml` with `extra="forbid"`.

## 🧪 Usage

```python
from open_food_mlops.config.settings import settings
from open_food_mlops.config.features import FEATURE_COLUMNS

print(settings.mlflow_tracking_uri)
assert "fat_100g" in FEATURE_COLUMNS
```

## ⚙️ Design Decisions

- **Path resolution in a model validator** — relative paths in `.env` become absolute against `base_dir`.
- **Production guardrails** — `localhost` in `MLFLOW_TRACKING_URI` raises when `APP_ENV=production`.
- **Feature list as `Final`** — prevents accidental mutation.

## 🚧 What Does NOT Belong Here

- Business logic (schemas only)
- YAML files (→ root `config/`)
- Secrets (→ `.env`, loaded by `settings.py`)

## 🔗 Related

- [Root config folder](../../../config/README.md)
- [Models subsystem](../models/README.md)