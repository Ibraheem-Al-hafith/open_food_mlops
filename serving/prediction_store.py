"""Thread-safe persistent storage interface for recording production predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import pandas as pd


class PredictionStore:
    """Handles thread-safe persistence of production prediction records to Parquet format."""

    def __init__(self, file_path: str | Path = "data/production/predictions.parquet") -> None:
        """Initialize the storage directory and file path.

        Args:
            file_path: Target path for storing production predictions Parquet dataset.
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        model_version: Optional[str] = None,
    ) -> None:
        """Append a prediction event alongside input features to storage.

        Args:
            features: Dictionary containing input feature key-value pairs.
            prediction: Resulting model prediction output (e.g. NOVA group).
            probability: Prediction confidence probability.
            model_version: Model identifier or version string.
        """
        record = {
            "timestamp": datetime.now(timezone.utc),
            **features,
            "prediction": prediction,
            "probability": probability,
            "model_version": model_version or "unknown",
        }

        new_data = pd.DataFrame([record])

        with self._lock:
            if self.file_path.exists():
                existing_df = pd.read_parquet(self.file_path)
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            else:
                combined_df = new_data

            combined_df.to_parquet(self.file_path, index=False)


# Singleton instance for default server-wide prediction recording
prediction_store = PredictionStore()