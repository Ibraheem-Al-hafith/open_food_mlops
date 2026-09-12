# 🛠️ Utils

> Cross-cutting utilities shared across the library.

## 🎯 Purpose

Small, dependency-free helpers that don't belong to any single subsystem.

## 🧩 Structure

```
utils/
├── logger.py   # setup_logging() — YAML-driven logging
└── README.md
```

## 🔌 Contract

`setup_logging(config_path=None)`:

- Loads `config/logging.yaml`
- Creates log directories if missing
- Removes pre-existing root handlers before applying config
- Falls back to `basicConfig(INFO)` if YAML is missing or invalid

## 🧪 Usage

```python
from open_food_mlops.utils.logger import setup_logging
setup_logging()  # call ONCE at process start
```

## ⚙️ Design Decisions

- **Idempotent** — safe to call multiple times (removes old handlers first).
- **Non-fatal** — a broken logging config never crashes the process.
- **Directory creation** — log rotation won't fail on first run.

## 🚧 What Does NOT Belong Here

- Domain logic (put it in the relevant subsystem)
- Anything that imports from `models/`, `data/`, `features/`, or `experiments/`

## 🔗 Related

- [Logging config](../../../config/logging.yaml)
- [Settings](../config/settings.py)
