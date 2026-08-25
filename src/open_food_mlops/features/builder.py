"""Feature-engineering pipeline construction.

This module is responsible for assembling the project's feature-engineering
pipeline.

Concrete transformers are intentionally kept outside this module. The
builder should answer:

    "Which feature transformations compose the production feature pipeline?"

It should not contain feature-engineering implementation details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import FeaturePipeline

if TYPE_CHECKING:
    import pandas as pd


def get_feature_pipeline() -> FeaturePipeline:
    """Build and return the project's feature-engineering pipeline.

    The concrete text, nutrition, and ingredient transformers will be added
    here as they are implemented.

    Returns:
        Configured ``FeaturePipeline`` ready to be fitted.

    Example:
        >>> pipeline = get_feature_pipeline()
        >>> X_engineered = pipeline.fit_transform(X_train)
        >>> X_test_engineered = pipeline.transform(X_test)
    """
    transformers = []

    # Concrete transformers will be added here:
    #
    # transformers.extend(
    #     [
    #         TextFeatureTransformer(...),
    #         NutritionFeatureTransformer(...),
    #         IngredientFeatureTransformer(...),
    #     ]
    # )

    if not transformers:
        raise RuntimeError(
            "The feature pipeline has no configured transformers. "
            "Add feature transformers before calling get_feature_pipeline()."
        )

    return FeaturePipeline(transformers)