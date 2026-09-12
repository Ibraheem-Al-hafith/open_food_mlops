# 📥 Data

> Ingestion, cleaning, and dataset splitting.

## 🎯 Purpose

Transform raw Open Food Facts CSV into a clean, stratified, Parquet dataset ready for training — without loading the full ~10 GB file into memory.

## 🧩 Structure

```
data/
├── data_ingestor.py   # Download + chunked processing
├── splitting.py       # Stratified train/test + K-fold
└── README.md
```

## 🔌 Contract

- **Input**: Open Food Facts TSV (gzipped) or a compatible CSV
- **Output**: `data/processed/processed_data.parquet`
- **Target column**: `nova_group` (0-indexed internally: `{0,1,2,3}` → public `{1,2,3,4}`)
- **Features**: exactly `FEATURE_COLUMNS` from `config/features.py`
- **Idempotency**: `.success` flag files short-circuit repeated runs

## 🧪 Usage

```python
from open_food_mlops.data.data_ingestor import OpenFoodFactsDataIngestor, DataConfig

ingestor = OpenFoodFactsDataIngestor(DataConfig())
df = ingestor.run()
```

Splitting:

```python
from open_food_mlops.data.splitting import DatasetSplits, DataSplitConfig

splits = DatasetSplits.from_dataframe(df, target="nova_group")
for fold in splits.splits:
    ...  # X_train, y_train, X_validation, y_validation
```

## ⚙️ Design Decisions

- **Chunked processing** — reads in 500k-row chunks; concatenates at the end.
- **`product_code` preserved** — needed for delayed ground-truth joins in monitoring.
- **Success flags** — safe to re-run; prevents redownloading 10 GB.
- **0-indexed targets internally** — sklearn convention; conversion happens at the serving boundary.

## 🚧 What Does NOT Belong Here

- Feature engineering → `features/`
- Model training → `experiments/`
- Batch inference → `pipelines/`

## 🔗 Related

- [Feature schema](../config/features.py)
- [Orchestrator](../experiments/orchestrator.py)
