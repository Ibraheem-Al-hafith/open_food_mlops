"""Thread-safe and concurrency-safe SQLite persistence for production predictions."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import pandas as pd

from open_food_mlops.config.settings import settings

logger = logging.getLogger(__name__)


class PredictionStore:
    """Handles persistent storage of production predictions using SQLite for concurrency safety."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize SQLite database schema and connection locks."""
        raw_path = Path(db_path or settings.predictions_path)
        if raw_path.suffix != ".db":
            raw_path = raw_path.with_suffix(".db")

        self.db_path = raw_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create prediction records table and indices if non-existent."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    product_code TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    features TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_product_code ON predictions(product_code)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)"
            )
            conn.commit()

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        model_version: Optional[str] = None,
        product_code: Optional[str] = None,
    ) -> None:
        """Append a single prediction record thread-safely."""
        sanitized_features = {k: v for k, v in features.items() if k != "product_code"}
        ts_str = datetime.now(timezone.utc).isoformat()
        p_code = product_code or "unknown"
        if not model_version:
            raise ValueError(
                "model_version is required when recording a prediction."
            )

        m_version = model_version
        features_json = json.dumps(sanitized_features)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO predictions (
                        timestamp, product_code, prediction, probability, model_version, features
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (ts_str, p_code, prediction, probability, m_version, features_json),
                )
                conn.commit()
            except Exception as exc:
                logger.error(
                    "Failed persisting prediction record: %s",
                    exc,
                    exc_info=True,
                )
                raise

    def read_predictions_dataframe(self) -> pd.DataFrame:
        """Read all recorded predictions into a pandas DataFrame."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            if not self.db_path.exists():
                return pd.DataFrame()
            df = pd.read_sql_query("SELECT * FROM predictions", conn)

        if df.empty:
            return df

        # Unpack JSON features into flat DataFrame columns for downstream monitoring compatibility
        features_df = pd.DataFrame(df["features"].apply(json.loads).tolist())
        df = df.drop(columns=["features"]).join(features_df)
        return df


prediction_store = PredictionStore()