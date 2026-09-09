"""Production Model Performance Evaluation & Degradation Detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.metrics import f1_score

from open_food_mlops.config.settings import settings
from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)

DEGRADATION_THRESHOLD = 0.85


def push_performance_metrics(
    macro_f1: float,
    degradation_ratio: float,
    gateway_url: str = settings.pushgateway_url,
) -> None:
    """Push evaluation metrics to Prometheus Pushgateway."""
    registry = CollectorRegistry()
    Gauge("model_production_macro_f1", "Production Macro-F1 score", registry=registry).set(macro_f1)
    Gauge("model_degradation_ratio", "Ratio of current Macro-F1 over baseline", registry=registry).set(degradation_ratio)

    try:
        push_to_gateway(gateway_url, job="batch_performance_eval", registry=registry)
        logger.info("Successfully pushed performance metrics to Pushgateway.")
    except Exception as e:
        logger.error("Failed to push metrics to Pushgateway: %s", e)


def evaluate_production_performance(
    predictions_path: str = str(settings.predictions_path),
    ground_truth_path: str = str(settings.ground_truth_path),
    baseline_f1: float = 0.65,
    threshold: float = DEGRADATION_THRESHOLD,
    gateway_url: str = settings.pushgateway_url,
) -> Optional[float]:
    """Evaluate model performance against delayed ground truth labels."""
    pred_file = Path(predictions_path)
    truth_file = Path(ground_truth_path)

    if not pred_file.exists() or not truth_file.exists():
        raise FileNotFoundError("Missing prediction store or ground truth dataset.")

    prod_df = pd.read_parquet(pred_file)
    truth_df = pd.read_parquet(truth_file)

    if "product_code" not in prod_df.columns or "product_code" not in truth_df.columns:
        raise KeyError("Missing 'product_code' linkage key across datasets.")

    valid_prod = prod_df[prod_df["product_code"] != "unknown"].copy()
    valid_prod = valid_prod.sort_values("timestamp").drop_duplicates(subset=["product_code"], keep="last")

    merged_df = valid_prod.merge(
        truth_df[["product_code", "nova_group"]],
        on="product_code",
        how="inner",
    )

    if merged_df.empty:
        raise RuntimeError("No matching valid product_code records available for performance evaluation.")

    y_true = merged_df["nova_group"].astype(int)
    if y_true.max() > 3:
        y_true = y_true - 1

    y_pred = merged_df["prediction"].astype(int)
    if y_pred.max() > 3:
        y_pred = y_pred - 1

    prod_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_ratio = prod_macro_f1 / baseline_f1 if baseline_f1 > 0 else 0.0

    push_performance_metrics(prod_macro_f1, f1_ratio, gateway_url=gateway_url)

    if f1_ratio < threshold:
        logger.critical("ALERT: Performance degradation detected! Macro-F1 ratio: %.2f%%", f1_ratio * 100)
    
    return prod_macro_f1


if __name__ == "__main__":
    setup_logging()
    evaluate_production_performance()