"""Generate model predictions for the monitoring reference dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings
from pipelines.batch_inference_flow import (
    extract_batch_predictions,
    load_champion_model,
)


REFERENCE_PATH = settings.reference_data_path
OUTPUT_PATH = settings.reference_predictions_path

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def generate_reference_predictions() -> None:
    print("Loading reference dataset...")

    reference_df = pd.read_parquet(REFERENCE_PATH)

    print(f"Reference rows available: {len(reference_df):,}")

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in reference_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if len(reference_df) > SAMPLE_SIZE:
        reference_df = reference_df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    feature_df = reference_df[FEATURE_COLUMNS]

    print(f"Rows selected for reference predictions: {len(feature_df):,}")

    print("Loading champion model...")

    model, run_id = load_champion_model(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    print(f"Champion model: {run_id}")
    print("Running inference...")

    raw_predictions, probabilities = extract_batch_predictions(
        model,
        feature_df,
    )

    predictions = np.where(
        raw_predictions < 4,
        raw_predictions + 1,
        raw_predictions,
    )

    output_df = pd.DataFrame(
        {
            "prediction": predictions.astype(int),
            "probability": probabilities.astype(float),
            "model_version": run_id,
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Reference predictions generated successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(output_df):,}")

    print()
    print("Prediction distribution:")
    print(
        output_df["prediction"]
        .value_counts()
        .sort_index()
    )


def main() -> None:
    generate_reference_predictions()


if __name__ == "__main__":
    main()
