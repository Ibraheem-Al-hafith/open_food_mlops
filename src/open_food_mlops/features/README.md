# 🧪 Features

> Composable, DataFrame-in / DataFrame-out feature transformers.

## 🎯 Purpose

Isolate feature engineering from models. Any transformer can be added,
removed, or reordered without touching training code.

## 🧩 Structure

```
features/
├── base.py                     # BaseFeatureTransformer + FeaturePipeline
├── builder.py                  # get_feature_pipeline() — the factory
├── identity.py                 # No-op transformer (baseline)
├── ADDING_A_TRANSFORMER.md     # 📘 Step-by-step tutorial
└── README.md
```

## 🔌 Transformer Contract (summary)

Every transformer **MUST**:
1. Inherit from `BaseFeatureTransformer`
2. Implement `_transform(X) -> pd.DataFrame`
3. Never mutate the input DataFrame in place
4. Return a DataFrame with stable column names

> 📘 **Full contract, templates, and golden rules:**
> [ADDING_A_TRANSFORMER.md](./ADDING_A_TRANSFORMER.md)

## 🧪 Minimal Usage

```python
from open_food_mlops.features.builder import get_feature_pipeline

pipeline = get_feature_pipeline()
X_train_t = pipeline.fit_transform(X_train)
X_test_t  = pipeline.transform(X_test)
```

## 🚧 What Does NOT Belong Here

- Model training → `models/`
- Data cleaning → `data/`
- Config loading → `config/`

## 🔗 Related

- 📘 [Adding a New Transformer](./ADDING_A_TRANSFORMER.md)
- [Feature schema](../config/features.py)
- [Models subsystem](../models/README.md)
