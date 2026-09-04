"""Registry for dynamically discoverable model implementations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TypeVar

from .base import BaseModel

logger = logging.getLogger(__name__)


ModelType = TypeVar("ModelType", bound=type[BaseModel])


_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register(
    name: str,
):
    """Register a model class under a unique name.

    Args:
        name: Stable configuration name used to identify the model.

    Returns:
        Class decorator.

    Raises:
        ValueError: If the name is empty or already registered.
    """
    normalized_name = name.strip().lower()

    if not normalized_name:
        raise ValueError("Model registry name cannot be empty.")

    def decorator(
        model_class: ModelType,
    ) -> ModelType:
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"{model_class.__name__} must inherit from BaseModel."
            )

        if normalized_name in _MODEL_REGISTRY:
            existing = _MODEL_REGISTRY[normalized_name]

            raise ValueError(
                f"Model {normalized_name!r} is already registered by "
                f"{existing.__module__}.{existing.__name__}."
            )

        _MODEL_REGISTRY[normalized_name] = model_class
        logger.info("Registered model %s from %s", normalized_name, model_class.__module__)

        return model_class

    return decorator


def get_model_class(
    name: str,
) -> type[BaseModel]:
    """Retrieve a registered model class.

    Args:
        name: Registered model name.

    Returns:
        Registered model class.

    Raises:
        KeyError: If the model is not registered.
    """
    normalized_name = name.strip().lower()

    try:
        model_class = _MODEL_REGISTRY[normalized_name]
        logger.debug("Resolved registered model class for %s: %s", name, model_class.__name__)
        return model_class
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<empty>"
        logger.error("Attempted to resolve unregistered model %r. Available: %s", name, available)

        raise KeyError(
            f"Model {name!r} is not registered. "
            f"Available models: {available}."
        ) from exc


def registered_models() -> Iterator[str]:
    """Return registered model names.

    Returns:
        Iterator over registered model names.
    """
    models = sorted(_MODEL_REGISTRY)
    logger.debug("Listing registered models: %s", models)
    yield from models