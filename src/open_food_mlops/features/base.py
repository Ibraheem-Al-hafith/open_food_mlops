"""Base abstractions for DataFrame-oriented feature engineering.

This module defines a small transformer interface inspired by the
scikit-learn transformer API.

The project deliberately keeps the interface DataFrame-oriented so that
feature names and column semantics remain available throughout the feature
engineering pipeline.

Core contract:

    fit(X, y=None)
    transform(X)
    fit_transform(X, y=None)
    get_feature_names_in()
    get_feature_names_out()

Concrete feature transformers such as text, nutrition, and ingredient
transformers should inherit from ``BaseFeatureTransformer``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Self

import pandas as pd


class BaseFeatureTransformer(ABC):
    """Base class for DataFrame-oriented feature transformers.

    The class intentionally follows the conceptual contract of
    ``sklearn.base.TransformerMixin`` while remaining independent from
    scikit-learn.

    Feature names are automatically inferred from the input and output
    DataFrames unless a subclass explicitly overrides the corresponding
    feature-name methods.

    Attributes:
        feature_names_in_: Names of the features observed during ``fit``.
        feature_names_out_: Names of the features produced by the fitted
            transformer.
        n_features_in_: Number of input features observed during ``fit``.
        n_features_out_: Number of output features produced by the fitted
            transformer.
        is_fitted_: Whether the transformer has successfully been fitted.
    """

    feature_names_in_: tuple[str, ...]
    feature_names_out_: tuple[str, ...]
    n_features_in_: int
    n_features_out_: int
    is_fitted_: bool

    def __init__(self) -> None:
        """Initialize the transformer state."""
        self.feature_names_in_ = ()
        self.feature_names_out_ = ()
        self.n_features_in_ = 0
        self.n_features_out_ = 0
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Self:
        """Fit the transformer to input data.

        Subclasses implement their actual fitting logic in
        ``_fit_transformer``.

        Args:
            X: Input feature DataFrame.
            y: Optional target Series. Most feature transformers do not need
                the target, but it is accepted to maintain compatibility with
                the scikit-learn transformer convention.

        Returns:
            The fitted transformer itself.

        Raises:
            TypeError: If ``X`` is not a pandas DataFrame.
            ValueError: If ``X`` has no columns or duplicate column names.
        """
        self._validate_input(X)

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        self._fit_transformer(X, y)

        self.is_fitted_ = True

        # Determine output feature names after fitting. Subclasses can
        # override get_feature_names_out() when automatic inference is not
        # sufficient.
        self.feature_names_out_ = self._infer_feature_names_out(X)

        self.n_features_out_ = len(self.feature_names_out_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data using the fitted transformer.

        Args:
            X: Input feature DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If the transformer has not been fitted.
            TypeError: If ``X`` is not a pandas DataFrame.
            ValueError: If required input features are missing.
        """
        self._check_is_fitted()
        self._validate_input(X)
        self._validate_feature_names(X)

        transformed = self._transform(X)

        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__}._transform() must return a "
                "pandas DataFrame."
            )

        return transformed

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit the transformer and transform the input data.

        This method follows the familiar scikit-learn transformer contract.

        Args:
            X: Input feature DataFrame.
            y: Optional target Series.

        Returns:
            Transformed DataFrame.
        """
        self.fit(X, y)

        transformed = self.transform(X)

        # The output schema is determined from the actual transformed
        # DataFrame. This is more reliable than trying to predict generated
        # feature names before transformation.
        if self._uses_automatic_output_feature_names:
            self.feature_names_out_ = tuple(
                str(column) for column in transformed.columns
            )
            self.n_features_out_ = len(self.feature_names_out_)

        return transformed

    def get_feature_names_in(self) -> tuple[str, ...]:
        """Return input feature names observed during fitting.

        Returns:
            Tuple containing input feature names.

        Raises:
            RuntimeError: If the transformer has not been fitted.
        """
        self._check_is_fitted()
        return self.feature_names_in_

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Return output feature names.

        Args:
            input_features: Optional input feature names. When supplied,
                they are validated against the feature names observed during
                fitting.

        Returns:
            Tuple containing output feature names.

        Raises:
            RuntimeError: If the transformer has not been fitted.
            ValueError: If ``input_features`` does not match the fitted input
                schema.
        """
        self._check_is_fitted()

        if input_features is not None:
            self._validate_input_feature_names(input_features)

        return self.feature_names_out_

    @property
    def _uses_automatic_output_feature_names(self) -> bool:
        """Whether output feature names should be inferred automatically.

        Subclasses can override this property and return ``False`` when they
        explicitly manage their output feature names.
        """
        return True

    def _fit_transformer(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
    ) -> None:
        """Fit transformer-specific state.

        Subclasses should override this method when fitting is required.

        The default implementation is a no-op because stateless transformers
        are valid feature transformers.

        Args:
            X: Input feature DataFrame.
            y: Optional target Series.
        """

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform transformer-specific feature engineering.

        Args:
            X: Input feature DataFrame.

        Returns:
            Transformed DataFrame.
        """
        raise NotImplementedError

    def _infer_feature_names_out(
        self,
        X: pd.DataFrame,
    ) -> tuple[str, ...]:
        """Infer output feature names from transformed data.

        The default implementation executes the transformation once in order
        to inspect the resulting DataFrame schema.

        Subclasses with expensive or stateful transformations should override
        this method and provide the output schema directly.

        Args:
            X: Input DataFrame.

        Returns:
            Inferred output feature names.
        """
        transformed = self._transform(X)

        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__}._transform() must return a "
                "pandas DataFrame."
            )

        return tuple(str(column) for column in transformed.columns)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Validate a transformer input DataFrame.

        Args:
            X: Input DataFrame.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If ``X`` is empty or has duplicate column names.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Feature transformers expect a pandas DataFrame, "
                f"got {type(X).__name__}."
            )

        if X.columns.empty:
            raise ValueError("Input DataFrame must contain at least one column.")

        if X.columns.has_duplicates:
            duplicates = X.columns[X.columns.duplicated()].unique().tolist()

            raise ValueError(
                "Input DataFrame contains duplicate column names: "
                f"{duplicates}."
            )

    def _validate_feature_names(self, X: pd.DataFrame) -> None:
        """Validate input features against the fitted schema.

        Extra columns are allowed because a transformer may intentionally
        select only a subset of the input DataFrame.

        Missing columns are rejected because they can make a fitted
        transformation invalid.

        Args:
            X: Input DataFrame.

        Raises:
            ValueError: If fitted input columns are missing.
        """
        missing = [
            feature
            for feature in self.feature_names_in_
            if feature not in X.columns
        ]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} is missing required input "
                f"features: {missing}."
            )

    def _validate_input_feature_names(
        self,
        input_features: Sequence[str],
    ) -> None:
        """Validate explicitly supplied input feature names.

        Args:
            input_features: Feature names supplied by the caller.

        Raises:
            ValueError: If names differ from fitted input features.
        """
        normalized = tuple(str(feature) for feature in input_features)

        if normalized != self.feature_names_in_:
            raise ValueError(
                "input_features does not match the features observed during "
                f"fit. Expected {self.feature_names_in_}, got {normalized}."
            )

    def _check_is_fitted(self) -> None:
        """Ensure the transformer has been fitted.

        Raises:
            RuntimeError: If ``fit()`` has not been called successfully.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                "Call fit() before using this method."
            )


class FeaturePipeline(BaseFeatureTransformer):
    """Compose multiple feature transformers sequentially.

    The pipeline behaves like a single feature transformer.

    Each transformer receives the DataFrame produced by the previous
    transformer:

        X
        ↓
        transformer_1
        ↓
        transformer_2
        ↓
        ...
        ↓
        transformed X

    This allows the training and serving layers to depend on one stable
    feature-engineering interface.

    Attributes:
        transformers: Ordered feature transformers.
    """

    def __init__(
        self,
        transformers: Iterable[BaseFeatureTransformer],
    ) -> None:
        """Initialize the feature pipeline.

        Args:
            transformers: Ordered collection of feature transformers.

        Raises:
            ValueError: If no transformers are supplied.
            TypeError: If an item is not a BaseFeatureTransformer.
        """
        super().__init__()

        self.transformers: tuple[
            BaseFeatureTransformer, ...
        ] = tuple(transformers)

        if not self.transformers:
            raise ValueError(
                "FeaturePipeline requires at least one transformer."
            )

        invalid = [
            transformer
            for transformer in self.transformers
            if not isinstance(transformer, BaseFeatureTransformer)
        ]

        if invalid:
            raise TypeError(
                "All pipeline components must inherit from "
                "BaseFeatureTransformer."
            )

    def _fit_transformer(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
    ) -> None:
        """Fit each transformer sequentially."""
        current = X

        for transformer in self.transformers:
            current = transformer.fit_transform(current, y)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transformers sequentially."""
        current = X

        for transformer in self.transformers:
            current = transformer.transform(current)

        return current

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit every transformer and return the fully transformed DataFrame.

        This implementation avoids an unnecessary second traversal of the
        pipeline by fitting and transforming each stage exactly once.

        Args:
            X: Input feature DataFrame.
            y: Optional target Series.

        Returns:
            Fully transformed DataFrame.
        """
        self._validate_input(X)

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        current = X

        for transformer in self.transformers:
            current = transformer.fit_transform(current, y)

        self.feature_names_out_ = tuple(
            str(column) for column in current.columns
        )
        self.n_features_out_ = len(self.feature_names_out_)
        self.is_fitted_ = True

        return current

    def _infer_feature_names_out(
        self,
        X: pd.DataFrame,
    ) -> tuple[str, ...]:
        """Infer output names from the final pipeline transformer.

        This method is not normally reached by ``fit_transform()``, which
        records the actual output schema directly.
        """
        current = X

        for transformer in self.transformers:
            current = transformer.transform(current)

        return tuple(str(column) for column in current.columns)

    @property
    def _uses_automatic_output_feature_names(self) -> bool:
        """Return whether output feature names are inferred automatically."""
        return True

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Return the final pipeline output feature names.

        Args:
            input_features: Optional input feature names to validate.

        Returns:
            Final engineered feature names.
        """
        return super().get_feature_names_out(input_features)

    def get_transformer(
        self,
        index: int,
    ) -> BaseFeatureTransformer:
        """Return a transformer at a given pipeline position.

        Args:
            index: Zero-based transformer index.

        Returns:
            Requested transformer.

        Raises:
            IndexError: If the index is outside the pipeline.
        """
        try:
            return self.transformers[index]
        except IndexError as exc:
            raise IndexError(
                f"Transformer index {index} is outside the pipeline."
            ) from exc