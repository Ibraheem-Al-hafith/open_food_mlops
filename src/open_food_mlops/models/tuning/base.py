"""Abstract hyperparameter optimization interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..base import BaseModel
from ..specs import SearchSpace


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Result returned by a hyperparameter optimization run."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int


class BaseTuner(ABC):
    """Abstract interface for hyperparameter optimization engines."""

    def __init__(
        self,
        model_class: type[BaseModel],
        search_space: SearchSpace,
        n_trials: int,
    ) -> None:
        """Initialize a tuner.

        Args:
            model_class: Model implementation being optimized.
            search_space: Hyperparameter search space.
            n_trials: Number of optimization trials.
        """
        self.model_class = model_class
        self.search_space = search_space
        self.n_trials = n_trials

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
    ) -> TuningResult:
        """Optimize model parameters."""