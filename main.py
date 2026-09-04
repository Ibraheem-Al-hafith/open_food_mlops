"""Main entry point driver for training pipelines and FastAPI serving."""

import argparse
import sys
import uvicorn

from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.experiments.orchestrator import ExperimentOrchestrator
from open_food_mlops.utils.logger import setup_logging
import yaml


def run_training(config_path: str) -> None:
    """Execute MLOps orchestrator experiment execution."""
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


def run_server(host: str, port: int) -> None:
    """Launch FastAPI serving REST endpoint."""
    uvicorn.run("serving.app:app", host=host, port=port, reload=False)


def main() -> None:
    """CLI routing entry point."""
    parser = argparse.ArgumentParser(description="Open Food MLOps Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train CLI command
    train_parser = subparsers.add_parser("train", help="Run ML training pipeline")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    # Serve CLI command
    serve_parser = subparsers.add_parser("serve", help="Run FastAPI serving app")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.config)
    elif args.command == "serve":
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()