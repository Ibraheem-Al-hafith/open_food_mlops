"""Fixtures for API tests. MLflow is only needed at import time, so tests need no server."""

from __future__ import annotations

import os
import tempfile

# Must happen before ``serving.app`` is imported: the module opens the prediction store.
os.environ.setdefault("PREDICTIONS_PATH", os.path.join(tempfile.mkdtemp(), "predictions.db"))
