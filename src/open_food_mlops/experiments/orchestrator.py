"""Core pipeline orchestrator coordinating data, training, tuning, and tracking."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.data.providers import TabularDataProvider
from open_food_mlops.evaluation.evaluator import Evaluator
from open_food_mlops.experiments.selection import (
    CandidateResult,
    ModelSelectionEngine,
    SelectionResult,
)
from open_food_mlops.features.builder import get_feature_pipeline
from open_food_mlops.models.factory import create_model
from open_food_mlops.models.registry import get_model_class
from open_food_mlops.models.tuning.optuna_tuner import OptunaTuner
from open_food_mlops.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Coordinates end-to-end MLOps pipeline execution."""

    def __init__(self, plan: ExperimentPlan) -> None:
        self.plan = plan
        self.tracker = MLflowTracker(
            tracking_uri=plan.tracking.tracking_uri,
            experiment_name=plan.tracking.experiment_name,
        )
        self.evaluator = Evaluator(
            primary_metric=plan.selection.primary_metric
        )
        self.selection_engine = ModelSelectionEngine(
            primary_metric=plan.selection.primary_metric,
            direction=plan.selection.direction,
            gates=plan.selection.gates,
        )

    def run(self) -> SelectionResult:
        """Executes full experiment pipeline across all enabled models."""
        data_provider = TabularDataProvider(
            data_path=self.plan.data.data_path,
            target_column=self.plan.data.target_column,
            # n_splits=self.plan.data.n_splits,
            # random_state=self.plan.data.random_state,
        )

        candidates: list[CandidateResult] = []

        for model_cfg in self.plan.models:
            if not model_cfg.enabled:
                logger.info("Skipping disabled model: %s", model_cfg.name)
                continue

            logger.info("Starting execution for model: %s", model_cfg.name)
            candidate = self._run_model_pipeline(model_cfg, data_provider)
            candidates.append(candidate)

        selection_result = self.selection_engine.select_champion(candidates)
        if selection_result.champion:
            logger.info(
                "Champion model selected: %s (ID: %s)",
                selection_result.champion.model_name,
                selection_result.champion.candidate_id,
            )
        else:
            logger.warning("No candidate passed quality gates.")

        return selection_result

    def _run_model_pipeline(
        self, model_cfg: Any, data_provider: TabularDataProvider
    ) -> CandidateResult:
        with self.tracker.start_run(run_name=f"{model_cfg.name}_run"):
            model_cls = get_model_class(model_cfg.name)
            best_params = model_cfg.params.copy()

            # 1. Hyperparameter Optimization
            if model_cfg.tuning.enabled:
                logger.info("Running HPO tuning for model %s", model_cfg.name)

                def objective(sampled_params: dict[str, Any]) -> float:
                    scores = []
                    for split in data_provider.get_splits():
                        feat_pipe = get_feature_pipeline()
                        X_tr = feat_pipe.fit_transform(split.X_train)
                        X_va = feat_pipe.transform(split.X_val)

                        m = model_cls({**model_cfg.params, **sampled_params})
                        m.fit(X_tr, split.y_train)
                        preds = m.predict(X_va)
                        score = self.evaluator.evaluate(split.y_val, preds).primary_score
                        scores.append(score)
                    return sum(scores) / len(scores)

                tuner = OptunaTuner(
                    model_class=model_cls,
                    search_space=model_cls.get_search_space(),
                    n_trials=model_cfg.tuning.trials,
                    direction=model_cfg.tuning.direction,
                    random_state=model_cfg.tuning.random_state,
                )
                tuning_res = tuner.optimize(objective)
                best_params.update(tuning_res.best_params)

            # 2. Final Training & Cross Validation Evaluation
            fold_metrics: list[dict[str, float]] = []
            for split in data_provider.get_splits():
                feat_pipe = get_feature_pipeline()
                X_tr = feat_pipe.fit_transform(split.X_train)
                X_va = feat_pipe.transform(split.X_val)

                model = model_cls(best_params)
                model.fit(X_tr, split.y_train)
                preds = model.predict(X_va)

                eval_res = self.evaluator.evaluate(split.y_val, preds)
                fold_metrics.append(eval_res.metrics)

            # Compute aggregated cross-validation metrics
            avg_metrics = {
                m: float(sum(f[m] for f in fold_metrics) / len(fold_metrics))
                for m in fold_metrics[0]
            }

            self.tracker.log_params(best_params)
            self.tracker.log_metrics(avg_metrics)

            # 3. Artifact Serialization
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