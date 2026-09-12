# 📓 Notebooks

> Exploratory analysis and maintenance notebooks.

## 🎯 Purpose

Scratch space for research, PoC work, and manual model maintenance. **Notebooks are not production code** — they are never imported, never tested, never scheduled.

## 🧩 Structure

| Notebook | Purpose |
|----------|---------|
| `01_ML_PoC.ipynb` | Initial proof-of-concept: EDA, baseline models |
| `02_model_maintainance.ipynb` | Manual model inspection, retraining experiments |

## 🔌 Contract

- Numbered prefixes indicate reading order.
- Every notebook must have a **Markdown title cell** explaining its purpose.
- Never commit outputs containing sensitive data or large files.
- If a notebook's logic becomes production-worthy → move it to `src/` and add tests.

## 🧪 Usage

```bash
jupyter lab notebooks/
```

## ⚙️ Design Decisions

- **Numbered prefixes** — keeps chronological order visible in the file tree.
- **Notebooks stay out of CI** — no execution during builds.

## 🚧 What Does NOT Belong Here

- Production logic (→ `src/open_food_mlops/`)
- Scheduled jobs (→ `pipelines/`)
- Tests (→ `tests/`)

## 🔗 Related

- [Pipelines](../pipelines/README.md)
- [Core library](../src/open_food_mlops/README.md)
