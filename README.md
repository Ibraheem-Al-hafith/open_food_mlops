# 🍎 Open Food MLOps Platform

> End-to-end MLOps pipeline for predicting NOVA food processing groups from nutritional data — from ingestion to production serving, monitoring, and drift detection.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688)](https://fastapi.tiangolo.com/)
[![Prefect](https://img.shields.io/badge/Prefect-orchestration-4B4BFF)](https://www.prefect.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800)](https://grafana.com/)

---

## 🎯 Overview

**Open Food MLOps** is a production-grade machine learning platform that classifies food products into **NOVA groups (1–4)** based on their nutritional composition. It covers the full MLOps lifecycle:

- 📥 **Data ingestion** from Open Food Facts
- 🧪 **Feature engineering** with a composable transformer pipeline
- 🤖 **Multi-model training** (Decision Tree, Random Forest, Logistic Regression, XGBoost, LightGBM)
- 🔍 **Hyperparameter tuning** via Optuna
- 🏆 **Champion selection** with quality gates
- 📦 **Model registry & aliasing** through MLflow
- 🚀 **Real-time serving** with FastAPI
- 📊 **Monitoring** with Prometheus + Grafana
- 🌊 **Drift detection** with Evidently
- 🔁 **Batch inference & performance evaluation** pipelines
- ⏰ **Prefect orchestration** for scheduled monitoring

The platform is **containerized**, **reproducible**, and ready to deploy on a VPS.

---

## 📊 Results

Champion model is selected dynamically based on `macro_f1`. Quality gates enforce minimum thresholds before promotion.

| Metric | Gate | Notes |
|--------|------|-------|
| Accuracy | ≥ 0.50 | Minimum acceptable |
| Macro F1 | ≥ 0.45 | Primary selection metric |

> 📈 Live metrics are available in Grafana once deployed (see [Monitoring](#-monitoring)).

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Open Food Facts] --> B[Data Ingestion]
    B --> C[Feature Pipeline]
    C --> D[Model Training + Tuning]
    D --> E{Quality Gates}
    E -->|Pass| F[MLflow Registry]
    E -->|Fail| G[Rejected]
    F --> H[FastAPI Serving]
    H --> I[Prediction Store SQLite]
    I --> J[Evidently Drift Reports]
    J --> K[Prometheus Pushgateway]
    K --> L[Grafana Dashboards]
```

### Service topology (docker-compose)

| Service | Internal | External | Purpose |
|---------|----------|----------|---------|
| FastAPI | 8000 | 1601 | Model inference API |
| MLflow | 5000 | 1602 | Tracking + registry |
| Prometheus | 9090 | 1603 | Metrics scraping |
| Pushgateway | 9091 | 1604 | Batch metrics |
| Grafana | 3000 | 1605 | Dashboards |
| Nginx | 80 | 1606 | Reverse proxy |

---

## 📦 Installation

### Prerequisites

- Python **3.11+**
- Docker & Docker Compose
- `uv` (recommended) or `pip`

### Local setup

```bash
git clone https://github.com/Ibraheem-Al-hafith/open_food_mlops.git
cd open_food_mlops

# Create env & install deps
uv sync
# or: pip install -r pyproject.toml
```

### Environment variables

Create a `.env` at the project root:

```env
APP_ENV=development
MLFLOW_TRACKING_URI=http://localhost:1602
PUSHGATEWAY_URL=http://localhost:1604
```

> ⚠️ In production, `MLFLOW_TRACKING_URI` and `PUSHGATEWAY_URL` **must not** point to localhost (enforced by `settings.py`).

---

## 🚀 Usage

### 1. Ingest data

```bash
python pipelines/data_ingestion_flow.py
```

### 2. Train & select champion

```bash
python main.py train --config config/experiment.yaml
```

### 3. Serve predictions

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### 4. Batch inference

```bash
python pipelines/batch_inference_flow.py --sample-size 10000
```

### 5. Generate reference data (for monitoring)

```bash
python pipelines/create_reference_data.py
python pipelines/create_reference_predictions.py
```

### 6. Run monitoring checks

```bash
python pipelines/monitoring_flow.py
```

### 7. Full stack via Docker

```bash
docker compose up --build
```

Then visit:

- API docs → http://localhost:1601/docs
- MLflow → http://localhost:1606/mlflow/
- Grafana → http://localhost:1606/grafana/
- Prometheus → http://localhost:1603

---

## 📡 API Reference

### `POST /v1/predict`

```bash
curl -X POST http://localhost:1601/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_code": "3017620422003",
    "added-sugars_100g": 5.0,
    "fat_100g": 30.0,
    "proteins_100g": 6.0,
    "fruits-vegetables-legumes_100g": 0.0,
    "sodium_100g": 0.2,
    "salt_100g": 0.5,
    "energy-kcal_100g": 550.0,
    "carbohydrates_100g": 57.0,
    "water_100g": 2.0
  }'
```

**Response**

```json
{
  "nova_group": 4,
  "probability": 0.9231
}
```

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true, "model_version": "production" }
```

### `GET /metrics`

Prometheus exposition format.

---

## 📁 Project Structure

```
.
├── config/                 # YAML experiment & logging configs
├── docker/                 # Dockerfiles, nginx, prometheus configs
├── docker-compose.yml      # Full stack orchestration
├── main.py                 # CLI entrypoint (train / serve)
├── monitoring/             # Evidently drift reports
├── notebooks/              # PoC & maintenance notebooks
├── pipelines/              # Prefect/CLI pipelines
│   ├── data_ingestion_flow.py
│   ├── batch_inference_flow.py
│   ├── evaluate_performance.py
│   ├── monitoring_flow.py
│   ├── prediction_drift.py
│   └── training_flow.py
├── serving/                # FastAPI app, metrics, prediction store
├── src/open_food_mlops/    # Core library
│   ├── config/             # Settings & schemas
│   ├── data/               # Ingestor & splitting
│   ├── evaluation/         # Evaluator
│   ├── experiments/        # Orchestrator & selection
│   ├── features/           # Transformer pipeline
│   ├── models/             # Base, registry, implementations, tuning
│   ├── tracking/           # MLflow wrapper
│   └── utils/              # Logging
└── tests/                  # Unit + integration tests
```

---

## 🧪 Testing

```bash
pytest tests/unit -v
pytest tests/integration -v
```

---

## 📊 Monitoring

- **Feature drift** → `monitoring/evidently_report.py` → `evidently_dataset_drift`
- **Prediction drift** → `pipelines/prediction_drift.py` → `evidently_prediction_drift`
- **Performance degradation** → `pipelines/evaluate_performance.py` → `model_degradation_ratio`
- **Serving metrics** → `serving/metrics.py` → latency, throughput, prediction distribution

All metrics are pushed to **Pushgateway** and scraped by **Prometheus**, visualized in **Grafana** (`monitoring/dashboard.json`).

> 🔔 Alert threshold: `model_degradation_ratio < 0.85` triggers a critical log.

---

## 🗺️ Roadmap

- [x] Data ingestion & processing
- [x] Multi-model training with Optuna tuning
- [x] MLflow registry with `production` alias
- [x] FastAPI serving + Prometheus metrics
- [x] Evidently drift detection
- [x] Prefect monitoring flow
- [ ] **Prefect scheduled deployments on VPS**
- [ ] **CI/CD via GitHub Actions**
- [ ] **Enhanced NOVA score prediction pipeline** (text + ingredients features)
- [ ] Model card + Hugging Face weights publication
- [ ] Colab demo notebook

---

## ⚠️ Limitations

- NOVA labels are **coarse**; class imbalance exists across groups.
- Features rely solely on **nutritional values** — no ingredient text yet.
- Predictions should **not** be used as medical or dietary advice.
- Production monitoring requires **delayed ground truth** for performance evaluation.
- SQLite prediction store is suitable for **single-node** deployments.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Run `pytest tests/ -v`
4. Open a Pull Request

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 📬 Contact

Maintainer: **Ibrahim Alhafiz** · [ibraheem.d.alhafiz@gmail.com](ibraheem.d.alhafiz@gmail.com)

---

> 🍕 *Because knowing what's in your food shouldn't require a PhD.*
