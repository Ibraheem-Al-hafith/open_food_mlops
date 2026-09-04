"""Configuration models for the model subsystem."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

logger.debug("Loaded core model configuration definitions.")


class TuningConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str = "optuna"
    trials: int = Field(default=50, gt=0)
    direction: str = "maximize"
    random_state: int = 42


class ModelConfig(BaseModel):
    """Configuration for selecting and constructing a model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningConfig = Field(default_factory=TuningConfig)