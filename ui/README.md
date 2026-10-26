# NOVA classifier UI

Gradio front-end for the serving API. Self-contained: own image, own requirements, own tests.
It only talks to the API over HTTP (`UI_API_URL`).

```
ui/
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── food_ui/        # application package (python -m food_ui.app)
└── tests/          # unit tests, no ML dependencies needed
```

## Run locally

```bash
cd ui
pip install -r requirements.txt pytest
UI_API_URL=http://localhost:1051 python -m food_ui.app      # http://localhost:7860
pytest                                                       # UI tests only
```

## Settings (`UI_*` environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `UI_API_URL` | `http://fastapi:8000` | Serving API as seen from the UI container |
| `UI_ROOT_PATH` | empty | Proxy prefix, `/ui` behind nginx |
| `UI_MAX_BATCH_ROWS` | `5000` | Upload row limit (the API enforces the same maximum) |
| `UI_BATCH_CHUNK_SIZE` | `500` | Rows per API call and progress step |
| `UI_REQUEST_TIMEOUT_S` | `60` | Timeout per API call |
| `UI_AUTH_USERNAME` / `UI_AUTH_PASSWORD` | unset | Enable login |

## Feature contract

`food_ui/features.py` repeats the model's 14 feature names so this image stays small.
`tests/test_ui_feature_contract.py` in the **main project** fails if they diverge.
