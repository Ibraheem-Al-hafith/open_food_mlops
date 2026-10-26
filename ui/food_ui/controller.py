"""Event handlers of the UI.

``UiController`` holds no Gradio layout code; it only translates widget values to
service calls and back. Its collaborators are injected, which keeps it testable.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import pandas as pd

from food_ui.batch import (
    BatchInputError,
    BatchOutcome,
    build_template_frame,
    prepare_batch,
    read_table,
    run_batch as execute_batch,
)
from food_ui.client import ApiError, PredictionClient
from food_ui.config import UiSettings
from food_ui.consistency import check_plausibility
from food_ui.examples import EXAMPLE_PRODUCTS
from food_ui.features import FEATURE_KEYS, FEATURES, component_label
from food_ui.nova import (
    render_batch_summary,
    render_placeholder,
    render_prediction,
    render_status,
    render_warnings,
)

logger = logging.getLogger(__name__)

RESULT_PLACEHOLDER = "Enter the nutrition values per 100 g, then select Predict."
BATCH_PLACEHOLDER = "Upload a file and select Run batch to see results here."
_EXPORT_MAX_AGE_S = 3600.0


class UiController:
    """Connects Gradio widgets with the prediction service."""

    def __init__(self, client: PredictionClient, settings: UiSettings) -> None:
        self._client = client
        self._settings = settings
        self._export_dir = Path(tempfile.mkdtemp(prefix="nova-ui-"))
        self.template_path = self._write_template()

    # ----- header ---------------------------------------------------------

    def status_html(self) -> str:
        """Model readiness pill for the page header."""
        health = self._client.health()
        return render_status(health.healthy, health.model_version)

    # ----- single product -------------------------------------------------

    def example_names(self) -> list[str]:
        """Names offered in the example dropdown."""
        return list(EXAMPLE_PRODUCTS)

    def load_example(self, name: Optional[str]) -> list[Any]:
        """Values for ``[product_code, *features]``; unchanged when nothing is selected."""
        if not name or name not in EXAMPLE_PRODUCTS:
            return [gr.skip()] * (1 + len(FEATURE_KEYS))
        product = EXAMPLE_PRODUCTS[name]
        return [name, *(product[key] for key in FEATURE_KEYS)]

    def clear(self) -> list[Any]:
        """Reset the form: ``[product_code, *features, result_html, warnings_html]``."""
        return ["", *([0] * len(FEATURE_KEYS)), render_placeholder(RESULT_PLACEHOLDER), ""]

    def predict_online(self, product_code: Optional[str], *values: Optional[float]) -> tuple[str, str]:
        """Classify one product; returns ``(result_html, warnings_html)``."""
        missing = [component_label(spec) for spec, value in zip(FEATURES, values) if value is None]
        if missing:
            raise gr.Error(f"Enter a value (0 if none) for: {', '.join(missing)}.")

        features = {spec.key: float(value) for spec, value in zip(FEATURES, values)}  # type: ignore[arg-type]
        warnings = render_warnings(check_plausibility(features))

        try:
            result = self._client.predict(features, (product_code or "").strip() or None)
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        return render_prediction(result.nova_group, result.probability), warnings

    # ----- batch ------------------------------------------------------------

    def run_batch(
        self,
        file_path: Optional[str],
        persist: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, pd.DataFrame, Optional[str]]:
        """Run an uploaded file; returns ``(summary_html, results_table, export_path)``."""
        if not file_path:
            raise gr.Error("Upload a CSV, TSV or Parquet file first.")

        try:
            progress(0.0, desc="Reading file")
            prepared = prepare_batch(read_table(file_path), self._settings.max_batch_rows)
            outcome = execute_batch(
                self._client,
                prepared,
                chunk_size=self._settings.batch_chunk_size,
                persist=persist,
                on_progress=lambda fraction, text: progress(fraction, desc=text),
            )
        except (BatchInputError, ApiError) as exc:
            raise gr.Error(str(exc)) from exc

        logger.info(
            "Batch finished: %d rows, %d predicted, %d rejected, persisted=%s",
            outcome.n_total, outcome.n_predicted, outcome.n_rejected, persist,
        )
        return self._summary(outcome), outcome.results, str(self._export(outcome.results))

    @staticmethod
    def _summary(outcome: BatchOutcome) -> str:
        return render_batch_summary(
            total=outcome.n_total,
            predicted=outcome.n_predicted,
            rejected=outcome.n_rejected,
            model_version=outcome.model_version,
            counts=outcome.group_counts(),
        )

    # ----- files ------------------------------------------------------------

    def _write_template(self) -> str:
        path = self._export_dir / "nova_batch_template.csv"
        build_template_frame().to_csv(path, index=False)
        return str(path)

    def _export(self, results: pd.DataFrame) -> Path:
        """Write results to a uniquely named CSV and prune stale exports."""
        self._prune_exports()
        path = self._export_dir / f"nova_predictions_{uuid.uuid4().hex[:8]}.csv"
        results.to_csv(path, index=False)
        return path

    def _prune_exports(self) -> None:
        cutoff = time.time() - _EXPORT_MAX_AGE_S
        for old in self._export_dir.glob("nova_predictions_*.csv"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except OSError:
                logger.debug("Could not remove stale export %s", old)
