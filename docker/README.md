# 🐳 Docker

> Dockerfiles and service configuration for the containerized platform.

## 🎯 Purpose

Encapsulates every service so the platform runs identically on a laptop, CI runner, or VPS. One `docker compose up` brings the entire stack online.

## 🧩 Structure

```
docker/
├── Dockerfile.fastapi       # Serving API image
├── Dockerfile.mlflow        # MLflow tracking + registry image
├── Dockerfile.pipeline      # Batch/training job image
├── nginx.conf               # Reverse proxy for /api, /mlflow, /grafana
├── prometheus.yml           # Scrape config for FastAPI + Pushgateway
└── mlflow_dockerization.md  # Notes on MLflow container setup
```

## 🔌 Contract

| Port | Public | Service |
|------|--------|---------|
| 8000 | 1601 | FastAPI |
| 5000 | 1602 | MLflow |
| 9090 | 1603 | Prometheus |
| 9091 | 1604 | Pushgateway |
| 3000 | 1605 | Grafana |
| 80 | 1606 | Nginx (proxy) |

## 🧪 Usage

```bash
# Full stack
docker compose up --build

# Single service
docker compose up fastapi
```

## ⚙️ Design Decisions

- **Nginx as single entrypoint** — one hostname, path-based routing (`/api/`, `/mlflow/`, `/grafana/`).
- **Healthchecks everywhere** — FastAPI and MLflow gate dependent services.
- **Named volumes for state** — `grafana_data`, `prometheus_data` survive rebuilds.
- **Pushgateway** — batch jobs that exit can still push metrics.

## 🚧 What Does NOT Belong Here

- Application code (mount or bake into images)
- Secrets (pass via `env_file: .env`)

## 🔗 Related

- [Root docker-compose.yml](../docker-compose.yml)
- [Monitoring folder](../monitoring/README.md)
