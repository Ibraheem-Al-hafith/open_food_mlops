"""Batch inference execution pipeline runner with canonical class mapping and sanitization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Tuple

import mlflow
import numpy as np
import pandas as pd

from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings
from open_food_mlops.utils.logger import setup_logging
from serving.prediction_store import PredictionStore

logger = logging.getLogger(__name__)


def load_champion_model(
    tracking_uri: str,
    experiment_name: str,
) -> Tuple[Any, str]:
    """Load champion model using MLflow model registry production alias, with fallback."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    alias_uri = "models:/open_food_champion@production"
    try:
        logger.info("Attempting loading champion model from registry alias: %s", alias_uri)
        model = mlflow.pyfunc.load_model(alias_uri)
        return model, "alias:production"
    except Exception as exc:
        logger.warning("Alias lookup failed for %s: %s. Falling back to run search.", alias_uri, exc)

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.macro_f1 DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(f"No finished runs found in MLflow experiment '{experiment_name}'.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    logger.info("Loading model fallback from run URI: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri), run_id


def extract_batch_predictions(
    model: Any, input_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute batch inference extracting predictions using estimator class definitions."""
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
        classes = getattr(estimator, "classes_", None)
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(input_df)
                if isinstance(probs, pd.DataFrame):
                    probs = probs.to_numpy()

                best_indices = np.argmax(probs, axis=1)
                max_probs = np.max(probs, axis=1)

                if classes is not None:
                    raw_preds = np.array([classes[idx] for idx in best_indices])
                else:
                    raw_preds = best_indices

                return raw_preds, max_probs
            except Exception as exc:
                logger.debug("Failed batch predict_proba on candidate %s: %s", estimator, exc)
                continue

    preds = model.predict(input_df)
    if isinstance(preds, (pd.DataFrame, pd.Series)):
        preds = preds.to_numpy().flatten()

    probabilities = np.ones_like(preds, dtype=float)
    return preds, probabilities


def run_batch_inference(
    data_path: str,
    output_store_path: str | Path | None = None,
    sample_size: int | None = None,
) -> None:
    """Read dataset, sanitize features, execute batch inference, and persist results."""
    logger.info("Starting batch inference pipeline...")

    if output_store_path is None:
        output_store_path = settings.predictions_path

    input_file = Path(data_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input batch data path not found: {data_path}")

    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)

    if sample_size and len(df) > sample_size:
        logger.info("Subsampling batch data to %d rows.", sample_size)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    product_codes = None
    if "product_code" in df.columns:
        product_codes = df["product_code"].astype(str).tolist()
    elif "code" in df.columns:
        product_codes = df["code"].astype(str).tolist()

    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        raise ValueError(f"Input batch dataset missing required features: {missing_features}")

    # Canonical Feature Extraction & Sanitization
    feature_df = df[FEATURE_COLUMNS].copy().astype(float)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model, run_id = load_champion_model(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    raw_preds, max_probs = extract_batch_predictions(model, feature_df)
    nova_groups = np.where(raw_preds < 4, raw_preds + 1, raw_preds)

    store = PredictionStore(db_path=output_store_path)

    logger.info("Persisting %d batch predictions to store %s...", len(feature_df), output_store_path)
    for idx, row in feature_df.iterrows():
        p_code = product_codes[idx] if product_codes else "unknown"
        store.record_prediction(
            features=row.to_dict(),
            prediction=int(nova_groups[idx]),
            probability=round(float(max_probs[idx]), 4),
            model_version=run_id,
            product_code=p_code,
        )

    logger.info("Batch inference completed successfully. Processed %d records.", len(feature_df))


def main() -> None:
    """CLI driver for batch inference pipeline."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Run MLOps Batch Inference Flow")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.ground_truth_path),
        help="Path to feature dataset for batch prediction.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional override for the SQLite prediction database path.",
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