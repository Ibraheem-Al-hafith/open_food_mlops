"""Batch inference execution pipeline runner using canonical data schema."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Tuple

import mlflow
import numpy as np
import pandas as pd

from open_food_mlops.config.settings import settings
from open_food_mlops.data.data_ingestor import DataConfig
from open_food_mlops.utils.logger import setup_logging
from serving.prediction_store import PredictionStore

logger = logging.getLogger(__name__)
data_config = DataConfig()
FEATURES = [f for f in data_config.features if f != data_config.target]


def load_champion_model(
    tracking_uri: str,
    experiment_name: str,
) -> Tuple[Any, str]:
    """Discover and load the champion model and its run ID from MLflow.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.

    Returns:
        Tuple containing loaded PyFunc MLflow model and run ID string.

    Raises:
        RuntimeError: If experiment or valid model artifact cannot be found.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.macro_f1 DESC"],
        max_results=5,
    )

    if not runs:
        raise RuntimeError(
            f"No finished runs found in MLflow experiment '{experiment_name}'."
        )

    for run in runs:
        run_id = run.info.run_id
        artifacts = client.list_artifacts(run_id)

        model_subpath = None
        for art in artifacts:
            if art.is_dir:
                dir_contents = [
                    f.path for f in client.list_artifacts(run_id, art.path)
                ]
                if any("MLmodel" in path for path in dir_contents):
                    model_subpath = art.path
                    break

        if not model_subpath:
            model_uri = (
                f"runs:/{run_id}"
                if any("MLmodel" in art.path for art in artifacts)
                else f"runs:/{run_id}/model"
            )
        else:
            model_uri = f"runs:/{run_id}/{model_subpath}"

        try:
            logger.info("Loading champion model from URI: %s", model_uri)
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            return loaded_model, run_id
        except Exception as exc:
            logger.warning("Failed loading model from URI %s: %s", model_uri, exc)
            continue

    raise RuntimeError(
        f"Unable to load a valid MLflow model artifact from top runs in '{experiment_name}'."
    )


def extract_batch_predictions(
    model: Any, input_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute batch inference and extract predictions and probabilities safely.

    Args:
        model: MLflow PyFunc or unwrapped estimator.
        input_df: DataFrame containing input features.

    Returns:
        Tuple of numpy arrays containing (predictions, probabilities).
    """
    candidate_estimators = []

    if hasattr(model, "unwrap_python_model"):
        try:
            unwrapped = model.unwrap_python_model()
            if unwrapped is not None:
                candidate_estimators.append(unwrapped)
        except Exception:
            pass

    model_impl = getattr(model, "_model_impl", None)
    if model_impl is not None:
        sklearn_model = getattr(model_impl, "sklearn_model", None)
        if sklearn_model is not None:
            candidate_estimators.append(sklearn_model)
        candidate_estimators.append(model_impl)

    candidate_estimators.append(model)

    for estimator in candidate_estimators:
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(input_df)
                if isinstance(probs, pd.DataFrame):
                    probs = probs.to_numpy()
                raw_preds = np.argmax(probs, axis=1)
                max_probs = np.max(probs, axis=1)
                return raw_preds, max_probs
            except Exception as exc:
                logger.debug("Failed batch predict_proba on candidate %s: %s", estimator, exc)
                continue

    preds = model.predict(input_df)
    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy().flatten()
    elif isinstance(preds, pd.Series):
        preds = preds.to_numpy()

    probabilities = np.ones_like(preds, dtype=float)
    return preds, probabilities


def run_batch_inference(
    data_path: str,
    output_store_path: str = "data/production/predictions.parquet",
    sample_size: int | None = None,
) -> None:
    """Read input dataset, extract canonical features, run inference, and persist predictions.

    Args:
        data_path: Path to raw or processed batch input dataset.
        output_store_path: Path to target parquet prediction store.
        sample_size: Optional row count limitation for batch processing.
    """
    logger.info("Starting batch inference pipeline...")
    input_file = Path(data_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input batch data path not found: {data_path}")

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if sample_size and len(df) > sample_size:
        logger.info("Subsampling batch data to %d rows.", sample_size)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Extract product codes if present before subsetting features
    product_codes = None
    if "product_code" in df.columns:
        product_codes = df["product_code"].astype(str).tolist()
    elif "code" in df.columns:
        product_codes = df["code"].astype(str).tolist()

    # Strictly enforce schema matching using canonical data features
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Input batch dataset missing required features: {missing_features}")

    feature_df = df[FEATURES].copy()

    model, run_id = load_champion_model(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    raw_preds, max_probs = extract_batch_predictions(model, feature_df)
    nova_groups = np.where(raw_preds < 4, raw_preds + 1, raw_preds)

    store = PredictionStore(file_path=output_store_path)

    logger.info("Persisting %d batch predictions to %s...", len(feature_df), output_store_path)
    for idx, row in feature_df.iterrows():
        p_code = product_codes[idx] if product_codes else "unknown"
        store.record_prediction(
            features=row.to_dict(),
            prediction=int(nova_groups[idx]),
            probability=round(float(max_probs[idx]), 4),
            model_version=run_id,
            product_code=p_code,
        )

    logger.info("Batch inference completed successfully. Generated %d predictions.", len(feature_df))


def main() -> None:
    """CLI driver for batch inference pipeline."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Run MLOps Batch Inference Flow")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/processed_data.parquet",
        help="Path to feature dataset for batch prediction.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/production/predictions.parquet",
        help="Target parquet prediction dataset path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sampling count.",
    )
    args = parser.parse_args()

    run_batch_inference(
        data_path=args.data_path,
        output_store_path=args.output_path,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()