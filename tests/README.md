## `tests/README.md`

# 🧪 Tests

> Unit and integration test suites.

## 🎯 Purpose

Guard the contracts of every module. Unit tests are fast and isolated; integration tests exercise the full orchestrator on a synthetic dataset.

## 🧩 Structure

```
tests/
├── unit/
│   ├── conftest.py                    # Shared fixtures (synthetic data)
│   ├── test_base.py                   # BaseModel contract
│   ├── test_components.py             # Evaluator + selection engine
│   ├── test_config.py                 # Pydantic schemas
│   ├── test_data_provider.py          # Data splitting
│   ├── test_factory.py                # create_model()
│   ├── test_model_implementation.py   # Every registered model
│   ├── test_optuna_tuner.py           # Tuning loop
│   ├── test_registry.py               # @register mechanics
│   └── test_specs.py                  # Search-space primitives
└── integration/
    └── test_pipeline.py               # End-to-end orchestrator run
```

## 🧪 Running

```bash
pytest tests/unit -v               # fast, isolated
pytest tests/integration -v        # slower, full pipeline
pytest tests/ -v                   # everything
```

## 🔌 Contract

- **Unit tests must not touch the network or the filesystem** beyond `tmp_path`.
- **Model tests are parametrized** — add a new model to the parametrize list, don't write a new test function.
- **Optional deps guarded** — `@pytest.mark.skipif(xgb is None, ...)` for xgboost/lightgbm/optuna.
- **Registry tests use a fixture** that backs up and restores `_MODEL_REGISTRY`.

## ➕ Adding Tests

- **New model** → append to the `parametrize` list in `test_model_implementation.py`
- **New transformer** → assert column stability across `fit_transform` / `transform`
- **New endpoint** → use FastAPI's `TestClient`
- **New pipeline** → add an integration test under `tests/integration/`

## ⚙️ Design Decisions

- **Synthetic data via `make_classification`** — no fixture files, fast, deterministic.
- **`conftest.py` centralizes fixtures** — no duplication across test files.
- **`tmp_path` for I/O** — pytest auto-cleans it.

## 🚧 What Does NOT Belong Here

- Notebooks → `notebooks/`
- Fixture files > 1 MB (generate synthetic data instead)
- Real datasets

## 🔗 Related

- [Models subsystem](../src/open_food_mlops/models/README.md)
- [Orchestrator](../src/open_food_mlops/experiments/orchestrator.py)
