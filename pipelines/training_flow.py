"""Training flow CLI pipeline runner."""

import argparse
import yaml
from src.open_food_mlops.config.schemas import ExperimentPlan
from src.open_food_mlops.experiments.orchestrator import ExperimentOrchestrator
from src.open_food_mlops.utils.logger import setup_logging


def run_pipeline(config_path: str) -> None:
    """Load experiment configuration and execute the orchestrator pipeline."""
    setup_logging()
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    plan = ExperimentPlan(**raw_config)
    orchestrator = ExperimentOrchestrator(plan)
    selection_result = orchestrator.run()

    if selection_result.champion:
        print(f"Pipeline Succeeded! Winner: {selection_result.champion.model_name}")
    else:
        print("Pipeline Execution Completed: No models passed quality gates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLOps Training Pipeline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config YAML"
    )
    args = parser.parse_args()
    run_pipeline(args.config)