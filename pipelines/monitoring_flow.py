"""Prefect orchestration flow for automated MLOps monitoring."""
from __future__ import annotations

import logging
from prefect import flow, task
from prefect.tasks import task_input_hash

# Import your existing pipeline functions
from pipelines.evaluate_performance import evaluate_production_performance
from monitoring.evidently_report import generate_drift_reports # Adjust if your function is named generate_drift_report
from pipelines.prediction_drift import generate_prediction_drift_report

logger = logging.getLogger(__name__)

@task(
    name="Evaluate Production Performance",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True
)
def task_evaluate_performance():
    """Task wrapper for performance evaluation with automatic retries."""
    print("🚀 Starting Production Performance Evaluation...")
    evaluate_production_performance()
    print("✅ Performance evaluation pushed to Pushgateway.")

@task(
    name="Check Feature Drift",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True
)
def task_check_feature_drift():
    """Task wrapper for Evidently feature drift."""
    print("🚀 Starting Feature Drift Analysis...")
    generate_drift_reports()
    print("✅ Feature drift metrics pushed to Pushgateway.")

@task(
    name="Check Prediction Drift",
    retries=2,
    retry_delay_seconds=10,
    log_prints=True
)
def task_check_prediction_drift():
    """Task wrapper for Evidently prediction drift."""
    print("🚀 Starting Prediction Drift Analysis...")
    generate_prediction_drift_report()
    print("✅ Prediction drift metrics pushed to Pushgateway.")

@flow(name="open-food-mlops-monitoring")
def monitoring_pipeline():
    """
    Main Prefect Flow that orchestrates the periodic ML monitoring checks.
    """
    print("="*50)
    print("🔍 EXECUTING OPEN FOOD MLOPS MONITORING PIPELINE")
    print("="*50)
    
    # Run tasks (Prefect handles the execution graph and retries)
    task_evaluate_performance()
    task_check_feature_drift()
    task_check_prediction_drift()
    
    print("="*50)
    print("🎉 ALL MONITORING CHECKS COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    # Running this file directly will execute the flow locally
    monitoring_pipeline()