from pathlib import Path

import numpy as np
import pandas as pd

from open_food_mlops.config.settings import settings
from pipelines.batch_inference_flow import (
    FEATURES,
    extract_batch_predictions,
    load_champion_model,
)


REFERENCE_PATH = Path("data/monitoring/reference.parquet")
OUTPUT_PATH = Path("data/monitoring/reference_predictions.parquet")

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def main() -> None:
    print("Loading reference dataset...")

    reference_df = pd.read_parquet(REFERENCE_PATH)

    print(f"Reference rows available: {len(reference_df):,}")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in reference_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Use a representative sample instead of running inference
    # over the entire 1.13M-row reference population.
    if len(reference_df) > SAMPLE_SIZE:
        reference_df = reference_df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    feature_df = reference_df[FEATURES]

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

    # Existing project convention:
    # raw classes 0-3 -> NOVA groups 1-4
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

    print()
    print("Prediction percentages:")
    print(
        (
            output_df["prediction"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )
    )

    print()
    print("Model versions:")
    print(output_df["model_version"].value_counts())


if __name__ == "__main__":
    main()