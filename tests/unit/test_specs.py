"""Unit tests for search space parameter specifications."""

from __future__ import annotations

import pytest

from open_food_mlops.models.specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)


def test_int_parameter_defaults() -> None:
    """Verify default values and structural immutability of IntParameter."""
    param = IntParameter(low=1, high=10)
    assert param.low == 1
    assert param.high == 10
    assert param.step == 1
    assert not param.log
    assert param.type == "int"

    with pytest.raises(AttributeError):
        param.low = 5  # type: ignore[misc]


def test_float_parameter_defaults() -> None:
    """Verify FloatParameter attributes and immutability."""
    param = FloatParameter(low=0.01, high=1.0, log=True)
    assert param.low == 0.01
    assert param.high == 1.0
    assert param.log
    assert param.type == "float"


def test_categorical_parameter_defaults() -> None:
    """Verify CategoricalParameter choice tuple mapping."""
    choices = ("gini", "entropy")
    param = CategoricalParameter(choices=choices)
    assert param.choices == choices
    assert param.type == "categorical"