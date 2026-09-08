"""Phase 7: Production Model Performance & Degradation Detection Pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.metrics import f1_score

from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)

PREDICTIONS_PATH = Path("data/production/predictions.parquet")
GROUND_TRUTH_PATH = Path("data/processed/processed_data.parquet")
TRAINING_BASELINE_F1 = 0.65
DEGRADATION_THRESHOLD = 0.85
PUSHGATEWAY_URL = "localhost:9091"


def push_performance_metrics(
    macro_f1: float,
    degradation_ratio: float,
    gateway_url: str = PUSHGATEWAY_URL,
) -> None:
    """Push batch evaluation metrics to Prometheus Pushgateway.

    Args:
        macro_f1: Calculated production Macro-F1 score.
        degradation_ratio: Calculated performance ratio against training baseline.
        gateway_url: Host and port of the Prometheus Pushgateway.
    """
    registry = CollectorRegistry()

    Gauge(
        "model_production_macro_f1",
        "Calculated production Macro-F1 score against delayed ground truth labels",
        registry=registry,
    ).set(macro_f1)

    Gauge(
        "model_degradation_ratio",
        "Ratio of current production Macro-F1 over baseline champion training Macro-F1",
        registry=registry,
    ).set(degradation_ratio)

    try:
        push_to_gateway(
            gateway_url, job="batch_performance_eval", registry=registry
        )
        logger.info("Successfully pushed performance metrics to Pushgateway.")
    except Exception as e:
        logger.error("Failed to push metrics to Pushgateway: %s", e)


def evaluate_production_performance(
    predictions_path: str = str(PREDICTIONS_PATH),
    ground_truth_path: str = str(GROUND_TRUTH_PATH),
    baseline_f1: float = TRAINING_BASELINE_F1,
    threshold: float = DEGRADATION_THRESHOLD,
    gateway_url: str = PUSHGATEWAY_URL,
) -> Optional[float]:
    """Evaluate production model performance against delayed ground truth data.

    Args:
        predictions_path: Path to production predictions parquet store.
        ground_truth_path: Path to canonical source of ground truth parquet.
        baseline_f1: Champion model baseline Macro-F1 score during training.
        threshold: Degradation alert threshold percentage (0.85 = 85%).
        gateway_url: Prometheus Pushgateway endpoint URL.

    Returns:
        Optional[float]: Calculated production Macro-F1 score if data exists.
    """
    pred_file = Path(predictions_path)
    truth_file = Path(ground_truth_path)

    if not pred_file.exists():
        logger.warning(
            "No production predictions parquet found at %s.", predictions_path
        )
        return None

    if not truth_file.exists():
        logger.warning("No ground truth dataset found at %s.", ground_truth_path)
        return None

    prod_df = pd.read_parquet(pred_file)
    truth_df = pd.read_parquet(truth_file)

    if (
        "product_code" not in prod_df.columns
        or "product_code" not in truth_df.columns
    ):
        logger.error(
            "Missing 'product_code' linkage in store/ground-truth tables."
        )
        return None

    merged_df = prod_df.merge(
        truth_df[["product_code", "nova_group"]],
        on="product_code",
        how="inner",
        suffixes=("_pred", "_true"),
    )

    if merged_df.empty:
        logger.info(
            "No matching ground-truth records found for current production predictions."
        )
        return None

    y_true = merged_df["nova_group_true"].astype(int)
    if y_true.max() > 3:
        y_true = y_true - 1

    y_pred = merged_df["prediction"].astype(int)
    if y_pred.max() > 3:
        y_pred = y_pred - 1

    prod_macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    f1_ratio = prod_macro_f1 / baseline_f1 if baseline_f1 > 0 else 0.0

    push_performance_metrics(
        prod_macro_f1, f1_ratio, gateway_url=gateway_url
    )

    logger.info(
        "Production Macro-F1: %.4f (Ratio to baseline: %.2f%%)",
        prod_macro_f1,
        f1_ratio * 100,
    )

    if f1_ratio < threshold:
        logger.critical(
            "ALERT: Model degradation detected! Production Macro-F1 ratio (%.2f%%) dropped below threshold (%.2f%%)",
            f1_ratio * 100,
            threshold * 100,
        )
    else:
        logger.info(
            "Model performance remains healthy within threshold limits."
        )

    return prod_macro_f1


def main() -> None:
    """CLI driver for performance evaluation."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate Production Model Performance"
    )
    parser.add_argument(
        "--predictions-path", type=str, default=str(PREDICTIONS_PATH)
    )
    parser.add_argument(
        "--ground-truth-path", type=str, default=str(GROUND_TRUTH_PATH)
    )
    parser.add_argument(
        "--baseline-f1", type=float, default=TRAINING_BASELINE_F1
    )
    parser.add_argument(
        "--threshold", type=float, default=DEGRADATION_THRESHOLD
    )
    parser.add_argument("--gateway-url", type=str, default=PUSHGATEWAY_URL)

    args = parser.parse_args()
    evaluate_production_performance(
        predictions_path=args.predictions_path,
        ground_truth_path=args.ground_truth_path,
        baseline_f1=args.baseline_f1,
        threshold=args.threshold,
        gateway_url=args.gateway_url,
    )


if __name__ == "__main__":
    main()