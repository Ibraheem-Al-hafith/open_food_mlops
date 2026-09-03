"""Optuna-backed hyperparameter optimization engine implementation."""

from __future__ import annotations

import logging
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler


from .base import BaseTuner, TuningResult
from ..base import BaseModel
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


class OptunaTuner(BaseTuner):
    """Hyperparameter optimizer using Optuna framework.

    Translates domain SearchSpace objects into Optuna trial suggestions and
    executes automated Bayesian optimization (TPE).
    """

    def __init__(
        self,
        model_class: type[BaseModel],
        search_space: SearchSpace,
        n_trials: int = 50,
        direction: str = "maximize",
        random_state: int | None = 42,
        show_progress_bar: bool = True,
    ) -> None:
        """Initialize the Optuna hyperparameter tuner.

        Args:
            model_class: Target model class definition.
            search_space: Mapping of hyperparameter specs to optimize.
            n_trials: Total trial budget for optimization.
            direction: Search goal; either 'maximize' or 'minimize'.
            random_state: Seed for sampler reproducibility.
            show_progress_bar: Flag to output Optuna progress bar.

        Raises:
            ImportError: If the 'optuna' library is not installed in the environment.
            ValueError: If 'direction' or 'n_trials' contains invalid values.
        """
        if optuna is None:
            raise ImportError(
                "Optuna package is not installed. Please install it via "
                "`pip install optuna` to run hyperparameter optimization."
            )

        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"Invalid optimization direction {direction!r}. "
                "Must be 'maximize' or 'minimize'."
            )

        if n_trials <= 0:
            raise ValueError(
                f"n_trials must be a positive integer, got {n_trials}."
            )

        super().__init__(
            model_class=model_class,
            search_space=search_space,
            n_trials=n_trials,
        )

        self.direction = direction
        self.random_state = random_state
        self.show_progress_bar = show_progress_bar

    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
    ) -> TuningResult:
        """Execute Optuna hyperparameter search loop.

        Args:
            objective: Callable accepting a hyperparameter dictionary and
                returning a validation performance score.

        Returns:
            TuningResult containing optimal hyperparameter values, optimal score,
            and actual trial count.
        """
        # Lower Optuna log level to maintain system logging standards
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )

        def _optuna_objective(trial: optuna.Trial) -> float:
            sampled_params = self._sample_params(trial, self.search_space)
            return objective(sampled_params)

        logger.info(
            "Starting Optuna search: model=%s, trials=%d, direction=%s",
            self.model_class.model_name,
            self.n_trials,
            self.direction,
        )

        study.optimize(
            _optuna_objective,
            n_trials=self.n_trials,
            show_progress_bar=self.show_progress_bar,
        )

        best_params = dict(study.best_params)
        best_score = float(study.best_value)

        logger.info(
            "Completed Optuna search: best_score=%.4f, best_params=%s",
            best_score,
            best_params,
        )

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
        )

    def _sample_params(
        self,
        trial: optuna.Trial,
        search_space: SearchSpace,
    ) -> dict[str, Any]:
        """Convert SearchSpace parameters to Optuna trial samples.

        Args:
            trial: Active Optuna Trial instance.
            search_space: Parameter space specification.

        Returns:
            Dictionary containing sampled parameter names and values.

        Raises:
            TypeError: If an unsupported search space parameter spec is encountered.
        """
        sampled_params: dict[str, Any] = {}

        for param_name, spec in search_space.items():
            sampled_params[param_name] = self._sample_single_param(
                trial=trial,
                name=param_name,
                spec=spec,
            )

        return sampled_params

    @staticmethod
    def _sample_single_param(
        trial: optuna.Trial,
        name: str,
        spec: SearchParameter,
    ) -> Any:
        """Map a single SearchParameter spec to Optuna trial suggestion methods."""
        if isinstance(spec, IntParameter):
            return trial.suggest_int(
                name=name,
                low=spec.low,
                high=spec.high,
                step=spec.step,
                log=spec.log,
            )

        if isinstance(spec, FloatParameter):
            return trial.suggest_float(
                name=name,
                low=spec.low,
                high=spec.high,
                log=spec.log,
            )

        if isinstance(spec, CategoricalParameter):
            return trial.suggest_categorical(
                name=name,
                choices=list(spec.choices),
            )

        raise TypeError(f"Unsupported search parameter spec type: {type(spec)!r}")