# Pipeline Configuration Reference Guide

This document provides a complete specification of all configuration parameters available in `config/experiment.yaml`. Use this guide as a reference when configuring data processing, feature engineering, model training, hyperparameter tuning, model selection, and tracking.

---

## Configuration Overview

The experiment configuration uses a declarative YAML structure parsed into Pydantic schema objects (`ExperimentPlan`). 

```yaml
name: open_food_classification_experiment
seed: 42

data:
  # Data handling and partitioning settings
features:
  # Feature transformation pipeline
models:
  # Model definitions and hyperparameter tuning
selection:
  # Champion model selection and quality gates
tracking:
  # Experiment tracking backend configuration

```

---

## Top-Level Fields

| Field | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `name` | `string` | N/A | **Yes** | Identifier for the experiment run. |
| `seed` | `integer` | `42` | No | Global pseudo-random seed for reproducibility across operations. |
| `data` | `object` | N/A | **Yes** | Data ingestion, sampling, and validation partitioning configuration. |
| `features` | `object` | Default pipeline | No | Feature engineering setup. |
| `models` | `list[object]` | N/A | **Yes** | List of model run specifications to train and evaluate. |
| `selection` | `object` | Default selection | No | Criteria and gates for champion model selection. |
| `tracking` | `object` | Default tracking | No | Artifact and metric logging configuration. |

---

## Section Details

### 1. Data (`data`)

Controls dataset ingestion paths, target label assignment, downsampling, and validation split strategies.

```yaml
data:
  data_path: data/processed/processed_data.parquet
  target_column: nova_group
  sample_fraction: 1.0
  test_size: 0.2
  n_splits: 5
  validation_method: stratified_kfold
  random_state: 42

```

#### Parameters

* **`data_path`** (`string`, **Required**)
* **Description**: File path to source dataset (`.csv` or `.parquet`).
* **Examples**: `"data/raw/food_sample.parquet"`, `"data/features.csv"`


* **`target_column`** (`string`, Default: `"nova_group"`)
* **Description**: Column name containing prediction target labels.


* **`sample_fraction`** (`float`, Default: `1.0`)
* **Description**: Stratified fraction of full dataset to sub-sample before splitting.
* **Range**: `0.0 < sample_fraction <= 1.0`
* **Usage**: Set to a lower value (e.g., `0.1`) for rapid prototyping.


* **`test_size`** (`float`, Default: `0.2`)
* **Description**: Proportion of dataset reserved for final holdout evaluation.
* **Range**: `0.0 < test_size < 1.0`


* **`validation_method`** (`string`, Default: `"stratified_kfold"`)
* **Description**: Strategy used for validation partitioning on training data.
* **Allowed Options**:
* `"stratified_kfold"`: Stratified $K$-Fold cross-validation (preserves class ratios across folds).
* `"kfold"`: Standard $K$-Fold cross-validation (unstratified).
* `"train_test_split"`: Single train/validation split.




* **`n_splits`** (`integer`, Default: `5`)
* **Description**: Number of folds for cross-validation strategies.
* **Constraints**: Required and active only when `validation_method` is `"stratified_kfold"` or `"kfold"`. Must be $\ge 2$.


* **`random_state`** (`integer`, Default: `42`)
* **Description**: Seed used for dataset sampling and splitting operations.



---

### 2. Features (`features`)

Defines data transformation steps applied to input features before model training.

```yaml
features:
  transformers:
    - identity

```

#### Parameters

* **`transformers`** (`list[string]`, Default: `["identity"]`)
* **Description**: Sequential list of feature transformer identifiers registered in feature pipeline.
* **Allowed Options**:
* `"identity"`: No-op transformer (passes features through unchanged).
* Custom registered transformer pipeline keys.





---

### 3. Models (`models`)

A list of candidate models to evaluate within the pipeline run.

```yaml
models:
  - name: decision_tree
    enabled: true
    params:
      max_depth: 10
      criterion: gini
    tuning:
      enabled: false
      method: optuna
      trials: 10
      direction: maximize
      random_state: 42

```

#### Model Item Fields

* **`name`** (`string`, **Required**)
* **Description**: Registered identifier of model implementation.
* **Allowed Options**:
* `"decision_tree"`: Decision Tree Classifier.
* `"random_forest"`: Random Forest Ensemble Classifier.




* **`enabled`** (`boolean`, Default: `true`)
* **Description**: Set to `false` to skip execution of this model without removing configuration block.


* **`params`** (`dict`, Default: `{}`)
* **Description**: Base hyperparameters passed directly to model constructor.
* **Supported Parameters by Model**:
* **`decision_tree`**:
* `max_depth` (`integer` or `null`, default: `null`)
* `criterion` (`string`, options: `["gini", "entropy", "log_loss"]`, default: `"gini"`)
* `min_samples_split` (`integer` or `float`, default: `2`)


* **`random_forest`**:
* `n_estimators` (`integer`, default: `100`)
* `max_depth` (`integer` or `null`, default: `null`)
* `criterion` (`string`, options: `["gini", "entropy", "log_loss"]`, default: `"gini"`)







#### Tuning Sub-block (`models[].tuning`)

Configures automated hyperparameter optimization using Optuna.

```yaml
tuning:
  enabled: true
  method: optuna
  trials: 20
  direction: maximize
  random_state: 42

```

* **`enabled`** (`boolean`, Default: `false`)
* **Description**: Enables hyperparameter search prior to final model evaluation.


* **`method`** (`string`, Default: `"optuna"`)
* **Description**: Search algorithm provider. Currently supports `"optuna"`.


* **`trials`** (`integer`, Default: `20`)
* **Description**: Number of search iterations (evaluations) to execute. Must be $> 0$.


* **`direction`** (`string`, Default: `"maximize"`)
* **Description**: Optimization goal for evaluation metric.
* **Allowed Options**: `"maximize"`, `"minimize"`


* **`random_state`** (`integer`, Default: `42`)
* **Description**: Random seed for hyperparameter sampling choices.



---

### 4. Selection (`selection`)

Defines criteria and automated quality threshold gates for selecting champion models.

```yaml
selection:
  primary_metric: macro_f1
  direction: maximize
  gates:
    accuracy: 0.50
    macro_f1: 0.45

```

#### Parameters

* **`primary_metric`** (`string`, Default: `"macro_f1"`)
* **Description**: Core metric used to rank candidate models.
* **Allowed Options**: `"accuracy"`, `"macro_f1"`, `"weighted_f1"`, `"micro_f1"`, `"precision"`, `"recall"`


* **`direction`** (`string`, Default: `"maximize"`)
* **Description**: Direction indicating a better score for `primary_metric`.
* **Allowed Options**: `"maximize"`, `"minimize"`


* **`gates`** (`dict[string, float]`, Default: `{}`)
* **Description**: Map of minimum required metric values. Models failing to pass *any* specified gate are disqualified from champion selection.
* **Example**:
```yaml
gates:
  accuracy: 0.60
  macro_f1: 0.55

```





---

### 5. Tracking (`tracking`)

Configures experiment tracking backend for logging parameters, metrics, and artifacts.

```yaml
tracking:
  backend: mlflow
  tracking_uri: sqlite:///mlflow.db
  experiment_name: open-food-mlops

```

#### Parameters

* **`backend`** (`string`, Default: `"mlflow"`)
* **Description**: Logging platform backend provider. Options: `"mlflow"`.


* **`tracking_uri`** (`string`, Default: `"sqlite:///mlflow.db"`)
* **Description**: Endpoint URI or local database connection path for MLflow tracking server.
* **Examples**: `"sqlite:///mlflow.db"`, `"http://localhost:5000"`, `"./mlruns"`


* **`experiment_name`** (`string`, Default: `"open-food-mlops"`)
* **Description**: Logical experiment grouping name within tracking backend.



---

## Validation Rules & Conditional Requirements

1. **Splitting Mechanics**:
* If `validation_method` is set to `"stratified_kfold"` or `"kfold"`, `n_splits` controls fold counts.
* If `validation_method` is `"train_test_split"`, single-split train/validation validation is used.


2. **Hyperparameter Tuning Rules**:
* When `models[].tuning.enabled` is `true`, hyperparameter bounds defined in `get_search_space()` for the given model will override static values configured in `models[].params`.
* Static values specified in `models[].params` remain active as defaults for non-tuned parameters.


3. **Quality Gate Filtering**:
* A candidate model must satisfy `metric_value >= gate_threshold` for all defined metrics in `selection.gates` when `direction` is `"maximize"`.
* If no trained candidate meets all defined `gates`, the selection engine returns `champion = None`.

