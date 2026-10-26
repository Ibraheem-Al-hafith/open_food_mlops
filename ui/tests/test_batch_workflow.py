from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import pytest

from food_ui.batch import BatchInputError, build_template_frame, prepare_batch, read_table, run_batch
from food_ui.client import BatchPrediction, BatchResult, BatchRowFailure
from food_ui.features import FEATURE_KEYS


class FakePredictor:
    """NOVA 4 when additives_n > 0, else NOVA 1; refuses product code 'bad'."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_batch(self, records: Sequence[dict[str, Any]], persist: bool = False) -> BatchResult:
        self.calls.append(len(records))
        predictions, errors = [], []
        for index, record in enumerate(records):
            if record.get("product_code") == "bad":
                errors.append(BatchRowFailure(index, "rejected by api"))
            else:
                group = 4 if record["additives_n"] > 0 else 1
                predictions.append(BatchPrediction(index, record.get("product_code"), group, 0.9))
        return BatchResult("production", tuple(predictions), tuple(errors))


def _frame(rows: int = 5) -> pd.DataFrame:
    frame = pd.concat([build_template_frame()] * rows, ignore_index=True)
    frame["product_code"] = [f"p{i}" for i in range(rows)]
    return frame


def test_template_has_expected_columns() -> None:
    assert list(build_template_frame().columns) == ["product_code", *FEATURE_KEYS]


def test_missing_columns_raise_with_names() -> None:
    with pytest.raises(BatchInputError, match="fat_100g"):
        prepare_batch(_frame().drop(columns=["fat_100g"]), max_rows=100)


def test_row_limit_is_enforced() -> None:
    with pytest.raises(BatchInputError, match="limit"):
        prepare_batch(_frame(5), max_rows=4)


def test_bad_rows_are_flagged_not_fatal() -> None:
    frame = _frame()
    frame.loc[1, "fat_100g"] = None
    frame.loc[2, "sugars_100g"] = -1
    frame.loc[3, "fruits-vegetables-legumes_100g"] = 150
    frame["water_100g"] = frame["water_100g"].astype(object)
    frame.loc[4, "water_100g"] = "n/a"

    prepared = prepare_batch(frame, max_rows=100)

    assert set(prepared.issues) == {1, 2, 3, 4}
    assert [pos for pos, _ in prepared.records] == [0]


def test_chunked_run_maps_results_back_to_original_rows() -> None:
    frame = _frame(5)
    frame.loc[1, "fat_100g"] = None          # rejected locally
    frame.loc[2, "product_code"] = "bad"      # rejected by the API
    frame.loc[3, "additives_n"] = 2           # NOVA 4
    predictor = FakePredictor()
    progress: list[float] = []

    outcome = run_batch(predictor, prepare_batch(frame, 100), chunk_size=2,
                        on_progress=lambda fraction, _: progress.append(fraction))

    results = outcome.results.set_index("row")
    assert predictor.calls == [2, 2]
    assert progress[-1] == 1.0
    assert (outcome.n_total, outcome.n_predicted, outcome.n_rejected) == (5, 3, 2)
    assert results.loc[1, "status"] == "predicted" and results.loc[1, "nova_group"] == 1
    assert results.loc[2, "status"] == "rejected" and "missing" in results.loc[2, "error"]
    assert results.loc[3, "error"] == "rejected by api"
    assert results.loc[4, "nova_group"] == 4
    assert outcome.group_counts() == {1: 2, 4: 1}


def test_all_rows_rejected_makes_no_api_call() -> None:
    frame = _frame(2)
    frame["fat_100g"] = None
    predictor = FakePredictor()

    outcome = run_batch(predictor, prepare_batch(frame, 100), chunk_size=10)

    assert predictor.calls == []
    assert outcome.n_predicted == 0 and outcome.n_rejected == 2


def test_read_table_keeps_leading_zeros_in_product_codes(tmp_path) -> None:
    path = tmp_path / "products.csv"
    frame = _frame(1)
    frame["product_code"] = ["0012345678905"]
    frame.to_csv(path, index=False)

    assert read_table(path)["product_code"].iloc[0] == "0012345678905"


def test_unsupported_suffix_raises(tmp_path) -> None:
    path = tmp_path / "products.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(BatchInputError, match="Unsupported"):
        read_table(path)
