"""Factory for constructing registered models."""

from __future__ import annotations

from typing import Any

from .base import BaseModel
from .config import ModelConfig
from .registry import get_model_class


def create_model(
    config: ModelConfig,
    **overrides: Any,
) -> BaseModel:
    """Create a model from configuration.

    Args:
        config: Model configuration.
        **overrides: Runtime parameter overrides, primarily useful during
            hyperparameter optimization.

    Returns:
        Configured model instance.

    Example:
        >>> model = create_model(config)
        >>> model.fit(X_train, y_train)
    """
    model_class = get_model_class(config.name)

    params = {
        **model_class.get_default_params(),
        **config.params,
        **overrides,
    }

    return model_class(params)