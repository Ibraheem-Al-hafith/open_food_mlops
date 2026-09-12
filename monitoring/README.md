# 📊 Monitoring

> Evidently-based drift detection and Grafana dashboard assets.

## 🎯 Purpose

Detect feature drift and data-quality issues in production by comparing
live predictions against a frozen reference dataset. Metrics are pushed
to Prometheus Pushgateway and visualized in Grafana.

## 🧩 Structure

```
monitoring/
├── evidently_report.py        # Feature drift report generator
├── dashboard.json             # Grafana dashboard definition
├── DASHBOARD_GUIDE.md         # 📘 How to import & read the dashboard
└── README.md
```

## 🔌 Contract

- **Inputs**: reference Parquet + current predictions (SQLite or Parquet)
- **Outputs**:
  - HTML report → `data/monitoring/reports/data_drift_<date>.html`
  - Prometheus metrics → `evidently_dataset_drift`, `evidently_share_of_drifted_columns`
- **Pushgateway job**: `batch_evidently_feature_drift`

## 🧪 Usage

```bash
python monitoring/evidently_report.py
```

## 🚧 What Does NOT Belong Here

- Prediction drift → `pipelines/prediction_drift.py`
- Performance evaluation → `pipelines/evaluate_performance.py`
- Serving metrics → `serving/metrics.py`

## 🔗 Related

- 📘 [Dashboard Guide](./DASHBOARD_GUIDE.md) — import + debugging mental model
- [Prometheus config](../docker/prometheus.yml)
- [Grafana dashboard](./dashboard.json)
