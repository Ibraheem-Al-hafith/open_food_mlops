"""Core orchestrator managing ingestion, feature engineering, model training, and selection."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.data.splitting import (
    DataSplitConfig,
    DatasetSplits,
    TestConfig,
    ValidationConfig,
    ValidationMethod,
)
from open_food_mlops.evaluation.evaluator import Evaluator
from open_food_mlops.experiments.selection import (
    CandidateResult,
    ModelSelectionEngine,
    SelectionResult,
)
from open_food_mlops.features.builder import get_feature_pipeline
import open_food_mlops.models.implementations  # Register models
from open_food_mlops.models.registry import get_model_class
from open_food_mlops.models.tuning.optuna_tuner import OptunaTuner
from open_food_mlops.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Coordinates end-to-end execution of MLOps pipelines."""

    def __init__(self, plan: ExperimentPlan) -> None:
        self.plan = plan
        self.tracker = MLflowTracker(
            tracking_uri=plan.tracking.tracking_uri,
            experiment_name=plan.tracking.experiment_name,
        )
        self.evaluator = Evaluator(primary_metric=plan.selection.primary_metric)
        self.selection_engine = ModelSelectionEngine(
            primary_metric=plan.selection.primary_metric,
            direction=plan.selection.direction,
            gates=plan.selection.gates,
        )

    def run(self) -> SelectionResult:
        """Execute experiment flow across all enabled models."""
        df = self._load_data(self.plan.data.data_path)

        split_config = DataSplitConfig(
            sample_fraction=self.plan.data.sample_fraction,
            test=TestConfig(
                test_size=self.plan.data.test_size,
                random_state=self.plan.data.random_state,
            ),
            validation=ValidationConfig(
                method=ValidationMethod(self.plan.data.validation_method),
                n_splits=self.plan.data.n_splits,
                random_state=self.plan.data.random_state,
            ),
        )

        dataset = DatasetSplits.from_dataframe(
            dataframe=df,
            target=self.plan.data.target_column,
            config=split_config,
        )

        candidates: list[CandidateResult] = []
        for model_cfg in self.plan.models:
            if not model_cfg.enabled:
                logger.info("Skipping disabled model: %s", model_cfg.name)
                continue

            try:
                candidate = self._run_model_pipeline(model_cfg, dataset)
                candidates.append(candidate)
            except Exception as err:
                logger.error("Failed executing model %s: %s", model_cfg.name, err, exc_info=True)

        return self.selection_engine.select_champion(candidates)

    def _load_data(self, path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _run_model_pipeline(
        self, model_cfg: Any, dataset: DatasetSplits
    ) -> CandidateResult:
        with self.tracker.start_run(run_name=f"{model_cfg.name}_run"):
            model_cls = get_model_class(model_cfg.name)
            best_params = model_cfg.params.copy()

            # Hyperparameter Optimization
            if model_cfg.tuning.enabled:
                def objective(sampled_params: dict[str, Any]) -> float:
                    scores = []
                    for split in dataset.splits:
                        pipe = get_feature_pipeline()
                        X_tr = pipe.fit_transform(split.X_train)
                        X_va = pipe.transform(split.X_validation)

                        m = model_cls({**model_cfg.params, **sampled_params})
                        m.fit(X_tr, split.y_train)
                        preds = m.predict(X_va)
                        scores.append(
                            self.evaluator.evaluate(split.y_validation, preds).primary_score
                        )
                    return float(sum(scores) / len(scores))

                tuner = OptunaTuner(
                    model_class=model_cls,
                    search_space=model_cls.get_search_space(),
                    n_trials=model_cfg.tuning.trials,
                    direction=model_cfg.tuning.direction,
                    random_state=model_cfg.tuning.random_state,
                )
                best_params.update(tuner.optimize(objective).best_params)

            # Cross Validation
            fold_metrics: list[dict[str, float]] = []
            for split in dataset.splits:
                pipe = get_feature_pipeline()
                X_tr = pipe.fit_transform(split.X_train)
                X_va = pipe.transform(split.X_validation)

                model = model_cls(best_params)
                model.fit(X_tr, split.y_train)
                preds = model.predict(X_va)

                fold_metrics.append(self.evaluator.evaluate(split.y_validation, preds).metrics)

            avg_metrics = {
                k: float(sum(f[k] for f in fold_metrics) / len(fold_metrics))
                for k in fold_metrics[0]
            }

            self.tracker.log_params(best_params)
            self.tracker.log_metrics(avg_metrics)

            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_dir = Path(tmp_dir) / model_cfg.name
                model.save(artifact_dir)
                self.tracker.log_artifact(str(artifact_dir))

                return CandidateResult(
                    candidate_id=f"candidate_{model_cfg.name}",
                    model_name=model_cfg.name,
                    metrics=avg_metrics,
                    params=best_params,
                    artifact_path=str(artifact_dir),
                )