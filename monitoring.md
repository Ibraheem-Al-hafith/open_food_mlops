Absolutely. We’ll do this in **two layers**:

1. First, I’ll give you a **high-level reference document** capturing the monitoring/observability architecture and how we arrived there.
2. Then we’ll immediately begin **Phase 1: API operational observability**, one small step at a time.

---

# Open Food MLOps — Monitoring & Observability Roadmap

## 1. Why We Are Adding Monitoring

The current project already has a solid ML lifecycle:

```text
Data
  ↓
Preprocessing
  ↓
Training
  ↓
Evaluation
  ↓
MLflow Experiment Tracking
  ↓
Model Registry
  ↓
Champion Model
  ↓
FastAPI Serving
```

The missing piece is **visibility after deployment**.

A model can be perfectly healthy during training and still encounter problems after deployment:

* API requests become slower.
* Error rates increase.
* Prediction traffic changes.
* Input data distribution changes.
* The model's real-world performance decreases.
* A particular feature starts behaving differently.
* Infrastructure problems affect model serving.

Therefore, monitoring is not simply "checking whether the API is alive."

The goal is to answer:

> **"Is the system healthy, and if it isn't, why?"**

---

# 2. The Monitoring Architecture

We decided to separate monitoring responsibilities instead of forcing one tool to do everything.

```text
                         ┌─────────────────────┐
                         │     FastAPI API      │
                         │                     │
                         │  /health            │
                         │  /predict           │
                         │  /metrics           │
                         └──────────┬──────────┘
                                    │
                           Operational Metrics
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Prometheus      │
                         │                     │
                         │ Time-series metrics │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Grafana        │
                         │                     │
                         │ Dashboards + Alerts │
                         └─────────────────────┘


       Training / Reference Data
                  │
                  ▼
        ┌───────────────────┐
        │     Evidently     │
        │                   │
        │ Data Drift        │
        │ Data Quality      │
        │ Model Performance │
        └─────────┬─────────┘
                  │
                  ▼
             Reports / Metrics


       Training & Model Lifecycle
                  │
                  ▼
        ┌───────────────────┐
        │      MLflow       │
        │                   │
        │ Experiments       │
        │ Training metrics  │
        │ Model artifacts   │
        │ Model registry    │
        └───────────────────┘


       Detailed debugging
                  │
                  ▼
             Application
                Logs
```

The important idea is:

> **Each tool has one primary responsibility.**

---

# 3. Responsibility of Each Tool

## MLflow — Model Lifecycle

MLflow answers:

> **"What happened when we trained and selected the model?"**

It already records things such as:

* experiments
* runs
* parameters
* metrics
* model artifacts
* registered models
* model versions

For example, the current training pipeline records `macro_f1`, accuracy, precision and recall, and uses `macro_f1` for model selection. The latest training output showed XGBoost being selected as the champion with a macro-F1 around `0.604`. 

MLflow therefore becomes our **training/reference source**.

---

# 4. Prometheus — Operational Monitoring

Prometheus answers:

> **"What is happening to the running service?"**

We will use it to monitor:

### API traffic

```text
requests / second
```

### Latency

```text
P50
P95
P99
```

### Errors

```text
4xx
5xx
```

### Prediction traffic

```text
predictions / second
predictions / minute
```

The FastAPI application will expose:

```text
GET /metrics
```

Prometheus periodically scrapes that endpoint.

Conceptually:

```text
FastAPI
   │
   │ /metrics
   ▼
Prometheus
   │
   ▼
time-series data
```

---

# 5. Grafana — Visualization and Alerting

Prometheus stores metrics, but it isn't intended to be our primary dashboard.

Grafana sits on top of Prometheus.

It will eventually provide dashboards such as:

```text
┌──────────────────────────────────────────┐
│        Open Food API Dashboard            │
├──────────────────────────────────────────┤
│                                          │
│ Request Rate       12.4 req/s            │
│                                          │
│ Error Rate          0.8 %                │
│                                          │
│ Prediction Rate     10.7 pred/s          │
│                                          │
│ P95 Latency         143 ms               │
│                                          │
├──────────────────────────────────────────┤
│ Request Rate History                     │
│       ╭──╮                               │
│  ─────╯  ╰───────────╮────              │
│                      ╰────              │
├──────────────────────────────────────────┤
│ Error Rate History                       │
│                                          │
└──────────────────────────────────────────┘
```

Later Grafana will also become our **alerting interface**.

For example:

```text
IF error_rate > 5%
FOR 5 minutes

→ ALERT
```

---

# 6. Evidently — ML Monitoring

Prometheus and Grafana are excellent for infrastructure and service metrics.

They are not the right tool for answering:

> "Has the distribution of our food features changed?"

That is where Evidently comes in.

We will eventually compare:

```text
Reference Data
(training / validation distribution)
          │
          │
          ▼
       Evidently
          ▲
          │
          │
Current Data
(recent production data)
```

Evidently will help us detect:

### Data drift

For example:

```text
fat_100g                    8% drift
sodium_100g                42% drift
energy-kcal_100g            5% drift
proteins_100g               3% drift
```

This can tell us that production inputs no longer resemble the data used during training.

---

# 7. Model Performance Monitoring

Because this project is a **multiclass classification problem**, MAE is not an appropriate primary metric.

Our primary metric is:

```text
macro_f1
```

with supporting metrics such as:

```text
accuracy
precision
recall
confusion matrix
per-class recall
```

The project already has an evaluation point where these metrics are computed. 

Eventually we want:

```text
Training / Reference Performance
                │
                ▼
          macro_f1 = 0.60
                │
                │
                │ comparison
                ▼
Recent Production Performance
                │
                ▼
          macro_f1 = 0.51
```

We can calculate:

```text
performance_ratio =
production_macro_f1 / reference_macro_f1
```

For example:

```text
0.51 / 0.60 = 0.85
```

Meaning production performance is approximately:

```text
15% lower
```

than the reference.

---

# 8. Data Drift + Model Performance

This is where monitoring becomes particularly useful.

We don't just want isolated numbers.

We want to **correlate signals**.

### Scenario A — Drift increases and F1 decreases

```text
Data Drift ↑
     +
Model F1 ↓
```

Possible explanation:

> The production data distribution changed, causing the model to perform worse.

---

### Scenario B — Drift stable but F1 decreases

```text
Data Drift →
Model F1 ↓
```

Possible causes include:

* concept drift
* labeling problems
* model issue
* changes in the relationship between features and target

---

### Scenario C — Drift increases but F1 remains stable

```text
Data Drift ↑
Model F1 →
```

The input distribution changed, but the model is still handling the change adequately.

---

### Scenario D — ML metrics stable but latency increases

```text
Data Drift →
Model F1 →
Latency ↑
```

This is probably **not an ML problem**.

It may be:

* CPU pressure
* memory pressure
* container problem
* network issue
* inefficient inference
* infrastructure degradation

This is exactly why we need both operational and ML monitoring.

---

# 9. The Final Monitoring Architecture

Eventually our system should look roughly like this:

```text
                         ┌──────────────────┐
                         │     Clients      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │                  │
                         │ /health          │
                         │ /predict         │
                         │ /metrics         │
                         └───┬──────────┬───┘
                             │          │
                 metrics     │          │ predictions
                             │          │
                             ▼          ▼
                       ┌──────────┐  ┌─────────────┐
                       │Prometheus│  │ Prediction  │
                       │          │  │ Data Store  │
                       └────┬─────┘  └──────┬──────┘
                            │                │
                            ▼                │
                       ┌──────────┐          │
                       │ Grafana  │          │
                       │Dashboard │          │
                       │+ Alerts  │          │
                       └──────────┘          │
                                             ▼
                                     ┌────────────────┐
                                     │    Evidently   │
                                     │                │
                                     │ Drift          │
                                     │ Data Quality   │
                                     │ Model Quality  │
                                     └───────┬────────┘
                                             │
                                             ▼
                                        Reports /
                                         Metrics


              ┌──────────────────────┐
              │        MLflow        │
              │                      │
              │ Training reference   │
              │ Experiments          │
              │ Models               │
              │ Registry             │
              └──────────────────────┘
```

---

# 10. How We Will Reach the Goal

We deliberately **will not build everything at once**.

That would make debugging the monitoring system itself unnecessarily difficult.

Instead, we will build vertical slices.

## Phase 1 — API Observability

Goal:

```text
FastAPI
   ↓
/metrics
```

We instrument:

* request count
* request latency
* HTTP status
* prediction count

At the end of Phase 1, the API itself knows how to expose useful metrics.

---

## Phase 2 — Prometheus

Add:

```text
FastAPI
   ↓
Prometheus
```

Prometheus periodically scrapes:

```text
http://api:8000/metrics
```

At this point we verify that metrics actually enter Prometheus.

---

## Phase 3 — Grafana

Add:

```text
Prometheus
     ↓
Grafana
```

Build the first dashboard:

* request rate
* error rate
* prediction rate
* P50/P95/P99 latency
* status distribution

---

## Phase 4 — Prediction Data

We need production prediction records before we can meaningfully monitor ML behavior.

Conceptually:

```text
Request
   ↓
Prediction
   ↓
record
   ↓
recent production dataset
```

This is also where the existing empty `batch_inference_flow.py` becomes important.

---

## Phase 5 — Evidently Data Drift

Introduce:

```text
Reference Dataset
        +
Current Dataset
        ↓
    Evidently
        ↓
Drift Report
```

Our processed dataset can serve as an initial reference source; the project currently stores `processed_data.parquet` under `data/processed`. 

---

## Phase 6 — Production Model Performance

Once ground truth becomes available:

```text
Recent Production Data
        ↓
Champion Model
        ↓
Predictions
        ↓
Ground Truth
        ↓
Evaluation
        ↓
macro_f1
```

Then:

```text
production macro_f1
          ↓
compare
          ↓
reference macro_f1
```

---

## Phase 7 — Alerts

Finally:

```text
Prometheus / Grafana
        +
Evidently
        +
Model Performance
        ↓
     Alerts
```

Examples:

```text
API error rate too high
        ↓
      ALERT
```

```text
P95 latency too high
        ↓
      ALERT
```

```text
Data drift too high
        ↓
      ALERT
```

```text
Production macro-F1 degraded
        ↓
      ALERT
```

---

# 11. The Most Important Design Principle

We are **not building a collection of dashboards**.

We are building a system that lets us move from:

```text
Something is wrong.
```

to:

```text
Something is wrong
        ↓
API latency increased
        ↓
Data drift also increased
        ↓
Model performance decreased
        ↓
The production data distribution probably changed
```

That is the real goal of observability.

---

# Phase 1 — Start Building

Now let's actually implement it.

Our first milestone is intentionally tiny:

```text
┌──────────────┐
│   FastAPI    │
│              │
│ /health      │
│ /predict     │
│ /metrics  ◄──┼── Prometheus later
└──────────────┘
```

**We are NOT adding Prometheus yet.**

First, we make sure the application can correctly produce its own metrics.

---

## Phase 1.1 — What We Are Going to Measure

We'll start with only three metric families.

### 1. Request counter

```text
open_food_api_requests_total
```

Example:

```text
GET /health → 200
POST /predict → 200
POST /predict → 500
```

Prometheus will see these as counts.

---

### 2. Request latency histogram

```text
open_food_api_request_duration_seconds
```

This lets us later calculate:

```text
P50
P95
P99
```

rather than just saying:

> "The average latency is 120 ms."

Histograms are much more useful for production latency monitoring.

---

### 3. Prediction counter

```text
open_food_predictions_total
```

This is specifically for:

```text
POST /predict
```

It tells us how much inference traffic the model is receiving.

We can optionally attach the predicted NOVA group as a label:

```text
nova_group="1"
nova_group="2"
nova_group="3"
nova_group="4"
```

These are low-cardinality values, so they're reasonable Prometheus labels.

We **will not** put things such as product names, IDs, request IDs, or arbitrary input values into Prometheus labels.

---

# Phase 1.2 — First File: `metrics.py`

I want to keep the instrumentation separate from the FastAPI application itself.

Create:

```text
serving/
├── app.py
├── schemas.py
└── metrics.py        ← new
```

Put this in `serving/metrics.py`:

```python
"""Prometheus metrics for the Open Food MLOps serving API."""

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "open_food_api_requests_total",
    "Total number of HTTP requests received by the API.",
    labelnames=("method", "endpoint", "status"),
)

REQUEST_LATENCY = Histogram(
    "open_food_api_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("method", "endpoint"),
)

PREDICTION_COUNT = Counter(
    "open_food_predictions_total",
    "Total number of predictions produced by the model.",
    labelnames=("nova_group",),
)
```

### Why this file?

We're establishing a clean separation:

```text
app.py
   │
   ├── API logic
   │
   └── metrics.py
           │
           └── observability definitions
```

The metrics themselves don't know anything about MLflow, Grafana, Docker or Evidently.

They are simply instruments.

---

# Phase 1.3 — Dependency

We also need the Prometheus Python client.

Because your project already uses `uv`, install it with:

```bash
uv add prometheus-client
```

Then verify:

```bash
uv run python -c "import prometheus_client; print(prometheus_client.__version__)"
```

Don't worry about Prometheus itself yet.

This is only the **Python instrumentation library**.

---

# Phase 1.4 — One Important Point Before Editing `app.py`

Your current FastAPI application is already nicely structured.

It has:

```text
app
 ├── lifespan
 ├── /health
 └── /predict
```

The `/predict` endpoint currently:

1. gets the champion model
2. converts the request into a DataFrame
3. performs prediction
4. converts the raw prediction into a NOVA group
5. returns the prediction and probability

This is visible directly in the existing serving code. 

So we **do not want to rewrite the inference logic**.

We'll simply add observability around it.

That distinction is important:

```text
Existing ML logic
        │
        │ DON'T TOUCH
        ▼
     prediction
        │
        ├──────────────► metric
        │
        ▼
      response
```

---

## Our immediate Phase 1 target

After the next few changes, this should work:

```bash
uv run python main.py serve
```

Then:

```text
http://localhost:8000/metrics
```

should return Prometheus-formatted output similar to:

```text
# HELP open_food_api_requests_total
# TYPE open_food_api_requests_total counter

open_food_api_requests_total{
    method="GET",
    endpoint="/health",
    status="200"
} 3
```

and:

```text
# HELP open_food_api_request_duration_seconds
# TYPE open_food_api_request_duration_seconds histogram
```

and:

```text
# HELP open_food_predictions_total
# TYPE open_food_predictions_total counter
```

**Once that works, Phase 1 is already halfway done.**

Then we'll connect:

```text
FastAPI → Prometheus
```

and only after that:

```text
Prometheus → Grafana
```

---

### Your next action

For now, **only do these two things**:

```bash
uv add prometheus-client
```

and create:

```text
serving/metrics.py
```

with the code above.

Then tell me the result (or paste any error), and we'll modify `serving/app.py` together. I want us to keep this incremental so that every layer is understood and verified before we build the next one.
