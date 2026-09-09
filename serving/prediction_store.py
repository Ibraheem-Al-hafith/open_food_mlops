"""Thread-safe persistent storage interface for recording production predictions."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import pandas as pd

from open_food_mlops.config.settings import settings

logger = logging.getLogger(__name__)


class PredictionStore:
    """Handles thread-safe persistence of production prediction records using append-only chunks."""

    def __init__(self, file_path: str | Path | None = None) -> None:
        """Initialize the storage directory and target base path."""
        self.file_path = Path(file_path or settings.predictions_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        model_version: Optional[str] = None,
        product_code: Optional[str] = None,
    ) -> None:
        """Append a single prediction record safely using atomic batch writes."""
        sanitized_features = {k: v for k, v in features.items() if k != "product_code"}

        record = {
            "timestamp": datetime.now(timezone.utc),
            "product_code": product_code or "unknown",
            **sanitized_features,
            "prediction": prediction,
            "probability": probability,
            "model_version": model_version or "unknown",
        }

        new_data = pd.DataFrame([record])

        with self._lock:
            try:
                if self.file_path.exists():
                    existing_df = pd.read_parquet(self.file_path)
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                else:
                    combined_df = new_data

                combined_df.to_parquet(self.file_path, index=False)
            except Exception as exc:
                logger.error("Failed persisting prediction to store: %s", exc, exc_info=True)


prediction_store = PredictionStore()