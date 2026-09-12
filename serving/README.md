# 🚀 Serving

> FastAPI inference layer with Prometheus instrumentation and durable prediction storage.

## 🎯 Purpose

Expose the champion model as a low-latency REST API, enforce strict input validation, and persist every prediction for downstream drift and performance monitoring.

## 🧩 Structure

```
serving/
├── app.py               # FastAPI app + lifespan model loading
├── schemas.py           # Pydantic request/response contracts
├── metrics.py           # Prometheus counters, histograms, middleware
├── prediction_store.py  # Thread-safe SQLite persistence
└── README.md
```

## 🔌 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model readiness (503 if unready) |
| POST | `/v1/predict` | Single-product NOVA classification |
| GET | `/metrics` | Prometheus scrape endpoint |

## 📡 Request / Response

```json
// POST /v1/predict
{
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
}

// Response
{ "nova_group": 4, "probability": 0.9231 }
```

## 🔌 Contract

- **Model loading**: `open_food_champion@production` alias, loaded once at startup.
- **NOVA normalization**: raw classes `0–3` → public groups `1–4`.
- **Validation**: NaN / inf rejected at the Pydantic boundary.
- **Persistence**: writes happen in a FastAPI `BackgroundTask`, never blocking the response.

## 🧪 Usage

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

## ⚙️ Design Decisions

- **Lifespan-based loading** — no cold start per request.
- **Estimator unwrapping chain** — handles `pyfunc → python_model → sklearn_model`.
- **SQLite + threading.Lock** — safe for single-node deployments; swap for Postgres at scale.
- **Fail-closed `/health`** — returns 503 if champion failed to load.

## 🚧 What Does NOT Belong Here

- Training logic → `src/open_food_mlops/experiments/`
- Batch inference → `pipelines/batch_inference_flow.py`
- Drift detection → `monitoring/`

## 🔗 Related

- [Prediction store schema](./prediction_store.py)
- [Prometheus config](../docker/prometheus.yml)