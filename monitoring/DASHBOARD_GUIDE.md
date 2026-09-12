# Open Food MLOps - Observability Dashboard Guide

This document explains how to import, read, and use the Unified Grafana Dashboard to monitor the operational and machine learning health of the Open Food Facts NOVA classification system.

## 1. How to Import the Dashboard

1. Ensure your Docker containers are running (`docker compose up -d`).
2. Open Grafana in your browser: `http://localhost:3000` (Default: `admin`/`admin`).
3. Verify your **Prometheus** data source is configured (`http://prometheus:9090`).
4. In the left menu, hover over **Dashboards** (four squares) and click **Import**.
5. Paste the JSON configuration into the "Import via panel json" text box and click **Load**.
6. Select your Prometheus data source from the dropdown and click **Import**.

---

## 2. How to Read the Dashboard (The 4 Zones)

The dashboard is divided into four distinct monitoring zones, answering different questions about your system.

### Zone 1: ML Health & Degradation (Top Left)
* **🚨 Model Degradation Ratio:** A gauge showing the current Production Macro-F1 divided by the Baseline Training Macro-F1. 
  * *Green:* > 85% (Healthy)
  * *Red:* < 85% (Alert: Model is degrading)
* **🎯 Production Macro-F1:** The exact calculated Macro-F1 score of the model against delayed ground-truth labels.

### Zone 2: Data Health & Drift (Top Right)
* **📊 Feature Drift Share:** The percentage of input features (e.g., `fat_100g`, `sodium_100g`) that have statistically drifted from the training baseline.
* **⚠️ Dataset Drift Flag:** A binary indicator (STABLE / DRIFT DETECTED) provided by Evidently's `DatasetDriftMetric`.

### Zone 3: Operational Health (Bottom Left)
* **📈 API Request Rate (RPS):** How much traffic the FastAPI server is handling.
* **❌ API Error Rate (5xx):** The percentage of requests resulting in internal server errors. Spikes here usually indicate code/schema bugs, not ML issues.

### Zone 4: Inference Performance (Bottom Right)
* **⏱️ P95 API Latency:** The time it takes for the model to return a prediction. Spikes here indicate infrastructure issues (memory leaks, MLflow timeouts).
* **📦 Prediction Volume:** The raw count of predictions being generated over time.

---

## 3. The Debugging Mental Model

When an issue occurs in production, use this decision tree to diagnose the root cause:

| Symptom | Dashboard Check | Diagnosis | Action Required |
| :--- | :--- | :--- | :--- |
| **"Predictions are slow"** | P95 Latency spikes, Request Rate is flat. | **Infrastructure Issue** | Check server memory, CPU, or MLflow connection timeouts. |
| **"API is failing"** | Error Rate (5xx) spikes. | **Code/Schema Bug** | Check FastAPI logs. A client likely sent a payload missing a required feature. |
| **"Model is making wrong predictions"** | Macro-F1 drops 🔴 + Feature Drift is HIGH 🔴 | **Covariate Shift** | The real world changed. Users are eating new foods the model never saw in training. **Retrain the model.** |
| **"Model is making wrong predictions"** | Macro-F1 drops 🔴 + Feature Drift is LOW 🟢 | **Concept Drift** | The inputs look normal, but the relationship to NOVA groups changed (or ground-truth labeling is broken). **Investigate data pipeline.** |
| **"Everything looks fine"** | Macro-F1 is STABLE 🟢 + Feature Drift is HIGH 🔴 | **Benign Drift** | The input data changed, but the model is robust enough to handle it. **Monitor closely.** |