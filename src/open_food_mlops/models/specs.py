"""Model hyperparameter search-space specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class IntParameter:
    """Integer hyperparameter search space."""

    low: int
    high: int
    step: int = 1
    log: bool = False

    type: Literal["int"] = "int"


@dataclass(frozen=True, slots=True)
class FloatParameter:
    """Floating-point hyperparameter search space."""

    low: float
    high: float
    log: bool = False

    type: Literal["float"] = "float"


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    """Categorical hyperparameter search space."""

    choices: tuple[Any, ...]

    type: Literal["categorical"] = "categorical"


SearchParameter: TypeAlias = (
    IntParameter
    | FloatParameter
    | CategoricalParameter
)

SearchSpace: TypeAlias = dict[str, SearchParameter]