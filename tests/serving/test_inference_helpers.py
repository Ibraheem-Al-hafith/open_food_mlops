import numpy as np
import pandas as pd
import pytest

from serving.inference import predict_with_confidence, to_nova_groups, validate_probabilities


class ProbaModel:
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile([0.1, 0.1, 0.7, 0.1], (len(frame), 1))


class LabelOnlyModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=int)


def test_predict_with_confidence_uses_probabilities() -> None:
    labels, confidences = predict_with_confidence(ProbaModel(), pd.DataFrame({"a": [1, 2]}))
    assert labels.tolist() == [2, 2] and confidences.tolist() == [0.7, 0.7]


def test_predict_falls_back_to_labels_with_full_confidence() -> None:
    labels, confidences = predict_with_confidence(LabelOnlyModel(), pd.DataFrame({"a": [1]}))
    assert labels.tolist() == [0] and confidences.tolist() == [1.0]


def test_to_nova_groups_maps_zero_based_classes() -> None:
    assert to_nova_groups(np.array([0, 1, 2, 3, 4])).tolist() == [1, 2, 3, 4, 4]


@pytest.mark.parametrize("bad", [[-1], [5]])
def test_to_nova_groups_rejects_out_of_range(bad: list[int]) -> None:
    with pytest.raises(ValueError):
        to_nova_groups(bad)


@pytest.mark.parametrize("bad", [[float("nan")], [1.5], [-0.2]])
def test_validate_probabilities_rejects_invalid(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_probabilities(bad)


def test_validate_probabilities_clips_float_noise() -> None:
    assert validate_probabilities([1.0 + 1e-9]).tolist() == [1.0]
