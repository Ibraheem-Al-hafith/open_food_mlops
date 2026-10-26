# Integrating the UI into open-food-mlops

Copy the folders of this package into your project root, keeping the paths.

## Where every file goes

| File | Action |
|---|---|
| `ui/` (whole folder) | **New.** Self-contained app: `Dockerfile`, `requirements.txt`, `food_ui/`, `tests/`, `README.md` |
| `serving/inference.py` | **New** |
| `serving/app.py` | **Replace** |
| `serving/schemas.py` | **Replace** |
| `pipelines/batch_inference_flow.py` | **Replace** |
| `docker/nginx.conf` | **Replace** (adds the `/ui/` route) |
| `tests/serving/` (3 files) | **New** |
| `tests/test_ui_feature_contract.py` | **New** |
| `docker-compose.yml` | **Edit** (below) |
| `.env` | **Edit** (below) |
| `pytest.ini` / `pyproject.toml` | **Edit**: add `ui` to `pythonpath` (below) |

## docker-compose.yml

Add the service (note the build context is the `ui/` folder itself):

```yaml
  ui:
    build:
      context: ./ui
    environment:
      UI_API_URL: http://fastapi:8000
      UI_ROOT_PATH: /ui
      UI_AUTH_USERNAME: ${UI_AUTH_USERNAME:-}
      UI_AUTH_PASSWORD: ${UI_AUTH_PASSWORD:-}
    depends_on:
      fastapi:
        condition: service_healthy
    restart: unless-stopped
```

and in the existing `nginx` service:

```yaml
    depends_on:
      ...
      ui:
        condition: service_healthy
```

## .env

```
UI_AUTH_USERNAME=choose-a-user
UI_AUTH_PASSWORD=choose-a-long-password
```

## Main project pytest config

The contract test imports `food_ui`, so add `ui` to the path:

```ini
[pytest]
pythonpath = . src ui
```

(or `pythonpath = [".", "src", "ui"]` under `[tool.pytest.ini_options]` in `pyproject.toml`).

## Deploy and test

```bash
docker compose up -d --build fastapi ui nginx     # UI at http://<vps-ip>:1606/ui/
pytest                                            # backend + contract tests (project root)
cd ui && pytest                                   # UI tests only
```

## Behaviour worth knowing

- `POST /v1/predict/batch` validates each record separately; invalid rows come back in `errors`.
- Batch predictions are stored only when `persist=true` (UI checkbox, off by default).
- The API stores `model_version="production"`; the CLI batch flow stores the MLflow run id. Align them if monitoring compares versions.
- On a public VPS, bind MLflow/Prometheus/Pushgateway ports to `127.0.0.1` so nginx is the only public entry point.
- Moving to a subdomain later: set `UI_ROOT_PATH` empty and give nginx a dedicated `server` block for the UI.
