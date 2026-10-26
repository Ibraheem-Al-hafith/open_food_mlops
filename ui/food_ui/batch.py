"""Batch workflow: read an uploaded table, validate it, run it through the API.

Everything here is independent of Gradio so it can be unit tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from food_ui.client import BatchResult
from food_ui.examples import EXAMPLE_PRODUCTS
from food_ui.features import FEATURE_KEYS, PRODUCT_CODE_COLUMN
from food_ui.nova import nova_label

FRUIT_VEG_KEY: Final[str] = "fruits-vegetables-legumes_100g"
CODE_FALLBACK_COLUMN: Final[str] = "code"
SUPPORTED_SUFFIXES: Final[tuple[str, ...]] = (".csv", ".tsv", ".parquet")
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "row", "product_code", "nova_group", "nova_label", "probability", "status", "error",
)
_MAX_LISTED_COLUMNS = 3

ProgressCallback = Callable[[float, str], None]


class BatchInputError(ValueError):
    """The uploaded file cannot be processed at all (as opposed to single bad rows)."""


class BatchPredictor(Protocol):
    """Anything able to run a batch call; implemented by ``PredictionClient``."""

    def predict_batch(
        self, records: Sequence[dict[str, Any]], persist: bool = False
    ) -> BatchResult: ...


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """Validated upload, split into rows ready to send and rows rejected locally."""

    total_rows: int
    product_codes: tuple[Optional[str], ...]
    records: tuple[tuple[int, dict[str, Any]], ...]
    issues: dict[int, str]


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Per-row results of a batch run, one row per uploaded row."""

    results: pd.DataFrame
    model_version: str

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_predicted(self) -> int:
        return int((self.results["status"] == "predicted").sum())

    @property
    def n_rejected(self) -> int:
        return self.n_total - self.n_predicted

    def group_counts(self) -> dict[int, int]:
        """Number of predicted products per NOVA group."""
        groups = self.results["nova_group"].dropna().astype(int)
        counts = groups.value_counts()
        return {int(str(group)): int(count) for group, count in counts.items()}


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV, TSV or Parquet file.

    Raises:
        BatchInputError: For unsupported or unreadable files.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    code_dtypes = {PRODUCT_CODE_COLUMN: "string", CODE_FALLBACK_COLUMN: "string"}

    try:
        if suffix == ".csv":
            return pd.read_csv(file_path, dtype=code_dtypes)
        if suffix == ".tsv":
            return pd.read_csv(file_path, sep="\t", dtype=code_dtypes)
        if suffix == ".parquet":
            return pd.read_parquet(file_path)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise BatchInputError(f"The file could not be read as {suffix[1:].upper()}: {exc}") from exc

    supported = ", ".join(SUPPORTED_SUFFIXES)
    raise BatchInputError(f"Unsupported file type '{suffix}'. Use one of: {supported}.")


def build_template_frame() -> pd.DataFrame:
    """One-row example table with the exact columns the batch tab expects."""
    row: dict[str, Any] = {PRODUCT_CODE_COLUMN: "example-apple"}
    row.update(EXAMPLE_PRODUCTS["Fresh apple"])
    return pd.DataFrame([row], columns=[PRODUCT_CODE_COLUMN, *FEATURE_KEYS])


def prepare_batch(frame: pd.DataFrame, max_rows: int) -> PreparedBatch:
    """Validate an uploaded table.

    File-level problems (missing columns, too many rows) raise ``BatchInputError``;
    row-level problems are collected in ``PreparedBatch.issues``.
    """
    table = frame.rename(columns=lambda name: str(name).strip())

    if PRODUCT_CODE_COLUMN not in table.columns and CODE_FALLBACK_COLUMN in table.columns:
        table = table.rename(columns={CODE_FALLBACK_COLUMN: PRODUCT_CODE_COLUMN})

    if table.empty:
        raise BatchInputError("The file contains no rows.")
    if len(table) > max_rows:
        raise BatchInputError(
            f"The file has {len(table):,} rows; the limit is {max_rows:,}. Split it and upload again."
        )

    missing = [key for key in FEATURE_KEYS if key not in table.columns]
    if missing:
        raise BatchInputError(
            f"Missing required columns: {', '.join(missing)}. Download the template for the expected format."
        )

    numeric = table[list(FEATURE_KEYS)].apply(pd.to_numeric, errors="coerce")
    issues = _find_row_issues(numeric)

    product_codes = _extract_product_codes(table)
    valid_positions = [pos for pos in range(len(table)) if pos not in issues]
    valid_rows = numeric.iloc[valid_positions].to_dict("records")

    records: list[tuple[int, dict[str, Any]]] = []
    for position, row in zip(valid_positions, valid_rows):
        record: dict[str, Any] = {key: float(value) for key, value in row.items()}
        if product_codes[position] is not None:
            record[PRODUCT_CODE_COLUMN] = product_codes[position]
        records.append((position, record))

    return PreparedBatch(
        total_rows=len(table),
        product_codes=product_codes,
        records=tuple(records),
        issues=issues,
    )


def _extract_product_codes(table: pd.DataFrame) -> tuple[Optional[str], ...]:
    if PRODUCT_CODE_COLUMN not in table.columns:
        return (None,) * len(table)

    codes: list[Optional[str]] = []
    for value in table[PRODUCT_CODE_COLUMN]:
        text = None if pd.isna(value) else str(value).strip()
        codes.append(text or None)
    return tuple(codes)


def _find_row_issues(numeric: pd.DataFrame) -> dict[int, str]:
    """Map row position to a message for rows that cannot be sent to the API."""
    missing = numeric.isna()
    infinite = pd.DataFrame(np.isinf(numeric.to_numpy()), columns=numeric.columns)
    negative = numeric < 0
    fruit_over = numeric[FRUIT_VEG_KEY] > 100

    flagged = missing.any(axis=1) | infinite.any(axis=1) | negative.any(axis=1) | fruit_over
    issues: dict[int, str] = {}

    for position in np.flatnonzero(flagged.to_numpy()):
        parts: list[str] = []
        if missing.iloc[position].any():
            parts.append(f"missing or non-numeric: {_columns(missing.iloc[position])}")
        if infinite.iloc[position].any():
            parts.append(f"infinite value: {_columns(infinite.iloc[position])}")
        if negative.iloc[position].any():
            parts.append(f"negative value: {_columns(negative.iloc[position])}")
        if fruit_over.iloc[position]:
            parts.append(f"{FRUIT_VEG_KEY} is above 100")
        issues[int(position)] = "; ".join(parts)

    return issues


def _columns(flags: pd.Series) -> str:
    names = [str(name) for name in flags.index[flags.to_numpy()]]
    shown = ", ".join(names[:_MAX_LISTED_COLUMNS])
    extra = len(names) - _MAX_LISTED_COLUMNS
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def run_batch(
    client: BatchPredictor,
    prepared: PreparedBatch,
    *,
    chunk_size: int,
    persist: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> BatchOutcome:
    """Send validated rows to the API in chunks and merge everything per row."""
    rows: dict[int, dict[str, Any]] = {
        position: _rejected_row(position, prepared.product_codes[position], detail)
        for position, detail in prepared.issues.items()
    }

    model_version = "n/a"
    records = prepared.records
    total_chunks = max(1, -(-len(records) // chunk_size))

    for chunk_number, start in enumerate(range(0, len(records), chunk_size), start=1):
        chunk = records[start : start + chunk_size]
        positions = [position for position, _ in chunk]

        result = client.predict_batch([record for _, record in chunk], persist=persist)
        model_version = result.model_version

        for prediction in result.predictions:
            position = positions[prediction.index]
            rows[position] = {
                "row": position + 1,
                "product_code": prepared.product_codes[position],
                "nova_group": prediction.nova_group,
                "nova_label": nova_label(prediction.nova_group),
                "probability": prediction.probability,
                "status": "predicted",
                "error": "",
            }
        for failure in result.errors:
            position = positions[failure.index]
            rows[position] = _rejected_row(position, prepared.product_codes[position], failure.detail)

        if on_progress is not None:
            on_progress(chunk_number / total_chunks, f"Predicted {min(start + chunk_size, len(records)):,} of {len(records):,} rows")

    for position in range(prepared.total_rows):
        rows.setdefault(
            position,
            _rejected_row(position, prepared.product_codes[position], "The service returned no result."),
        )

    ordered = [rows[position] for position in sorted(rows)]
    results = pd.DataFrame(ordered, columns=list(RESULT_COLUMNS))
    results["nova_group"] = results["nova_group"].astype("Int64")
    return BatchOutcome(results=results, model_version=model_version)


def _rejected_row(position: int, product_code: Optional[str], detail: str) -> dict[str, Any]:
    return {
        "row": position + 1,
        "product_code": product_code,
        "nova_group": None,
        "nova_label": "",
        "probability": None,
        "status": "rejected",
        "error": detail,
    }
