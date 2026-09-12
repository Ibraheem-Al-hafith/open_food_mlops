# 🔁 Pipelines

> Thin orchestration flows for ingestion, training, inference, and monitoring.

## 🎯 Purpose

Each file is an **independently runnable flow** that composes library code from `src/open_food_mlops/` and `serving/`. Pipelines are deliberately thin — they orchestrate, they don't implement.

## 🧩 Structure

| File | Trigger | Purpose |
|------|---------|---------|
| `data_ingestion_flow.py` | Manual / cron | Download + process Open Food Facts |
| `training_flow.py` | Manual / CI | Train models, select champion |
| `batch_inference_flow.py` | Scheduled | Score new product batches |
| `create_reference_data.py` | One-off | Freeze reference features for drift |
| `create_reference_predictions.py` | One-off | Freeze reference predictions for drift |
| `evaluate_performance.py` | Daily | Compare vs delayed ground truth |
| `prediction_drift.py` | Daily | Evidently prediction drift |
| `monitoring_flow.py` | Prefect | Orchestrates all monitoring tasks |

## 🔌 Contract

Every pipeline:
- Is runnable via `python pipelines/<file>.py`
- Has a `main()` guarded by `if __name__ == "__main__"`
- Loads settings from `open_food_mlops.config.settings`
- Pushes metrics to Pushgateway (if monitoring-related)
- Never contains business logic — imports it

## 🧪 Usage

```bash
# Ingestion
python pipelines/data_ingestion_flow.py

# Batch inference (10k rows)
python pipelines/batch_inference_flow.py --sample-size 10000

# Full monitoring suite
python pipelines/monitoring_flow.py
```

## ⚙️ Design Decisions

- **Prefect for monitoring only** — training stays CLI-driven for reproducibility.
- **Separate reference creation scripts** — reference data is generated *once* per champion.
- **Champion loaded via MLflow alias** — `models:/open_food_champion@production`, with fallback to best run.

## 🚧 What Does NOT Belong Here

- Business logic → `src/open_food_mlops/`
- HTTP endpoints → `serving/`
- Config schemas → `config/`

## 🔗 Related

- [Orchestrator](../src/open_food_mlops/experiments/orchestrator.py)
- [Monitoring folder](../monitoring/README.md)
