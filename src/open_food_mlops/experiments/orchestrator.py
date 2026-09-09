"""Orchestrates model pipeline execution, single-pass refit, packaging, and MLflow promotion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.config.settings import settings
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
import open_food_mlops.models.implementations  # Register implementations
from open_food_mlops.models.registry import get_model_class
from open_food_mlops.models.tuning.optuna_tuner import OptunaTuner
from open_food_mlops.models.wrapper import NovaPipelineWrapper
from open_food_mlops.tracking.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainedCandidate:
    """Internal container holding metadata and fitted artifacts for a candidate model."""

    result: CandidateResult
    fitted_pipeline: Any
    fitted_model: Any


class ExperimentOrchestrator:
    """Orchestrates cross-validation, hyperparameter tuning, final refit, and champion promotion."""

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
        """Execute experiment workflow and promote champion without duplicate computation."""
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

        trained_candidates: list[TrainedCandidate] = []
        for model_cfg in self.plan.models:
            if not model_cfg.enabled:
                logger.info("Skipping disabled model: %s", model_cfg.name)
                continue

            try:
                candidate = self._run_model_pipeline(model_cfg, dataset)
                trained_candidates.append(candidate)
            except Exception as err:
                logger.error("Failed executing model %s: %s", model_cfg.name, err, exc_info=True)

        candidate_results = [tc.result for tc in trained_candidates]
        selection_result = self.selection_engine.select_champion(candidate_results)

        if selection_result.champion:
            winning_candidate = next(
                tc for tc in trained_candidates if tc.result.candidate_id == selection_result.champion.candidate_id
            )
            self._promote_champion(winning_candidate)

        return selection_result

    def _load_data(self, path: str) -> pd.DataFrame:
        target_path = Path(path)
        if not target_path.is_absolute():
            target_path = settings.base_dir / target_path
        return pd.read_parquet(target_path) if target_path.suffix == ".parquet" else pd.read_csv(target_path)

    def _run_model_pipeline(
        self, model_cfg: Any, dataset: DatasetSplits
    ) -> TrainedCandidate:
        with self.tracker.start_run(run_name=f"{model_cfg.name}_run"):
            model_cls = get_model_class(model_cfg.name)
            best_params = model_cfg.params.copy()

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

            # Refit once on full training split
            logger.info("Refitting %s on full training dataset...", model_cfg.name)
            final_pipeline = get_feature_pipeline()
            X_train_full = final_pipeline.fit_transform(dataset.X_train)
            final_model = model_cls(best_params)
            final_model.fit(X_train_full, dataset.y_train)

            # Persist local artifacts using settings.base_dir
            artifacts_dir = settings.base_dir / "data" / "artifacts" / model_cfg.name
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            final_model.save(artifacts_dir)
            joblib.dump(final_pipeline, artifacts_dir / "feature_pipeline.joblib")
            self.tracker.log_artifact(str(artifacts_dir))

            candidate_res = CandidateResult(
                candidate_id=f"candidate_{model_cfg.name}",
                model_name=model_cfg.name,
                metrics=avg_metrics,
                params=best_params,
                artifact_path=str(artifacts_dir),
            )

            return TrainedCandidate(
                result=candidate_res,
                fitted_pipeline=final_pipeline,
                fitted_model=final_model,
            )

    def _promote_champion(self, champion_candidate: TrainedCandidate) -> None:
        """Bundle pipeline and champion model into MLflow PyFunc and set production alias."""
        logger.info(
            "Promoting champion model '%s' to MLflow Model Registry...",
            champion_candidate.result.model_name,
        )

        wrapper = NovaPipelineWrapper(
            pipeline=champion_candidate.fitted_pipeline,
            model=champion_candidate.fitted_model,
        )

        with self.tracker.start_run(run_name="champion_promotion"):
            self.tracker.log_params(champion_candidate.result.params)
            self.tracker.log_metrics(champion_candidate.result.metrics)

            self.tracker.register_and_alias_pyfunc(
                pyfunc_model=wrapper,
                artifact_path="model",
                registered_model_name="open_food_champion",
                alias="production",
            )
            logger.info("Champion bundled model successfully registered and aliased as 'open_food_champion@production'.")