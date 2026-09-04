"""Model selection engine applying quality gates and performance ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """Represents candidate performance evaluation from a model run."""

    candidate_id: str
    model_name: str
    metrics: dict[str, float]
    params: dict[str, Any]
    artifact_path: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Outcome of model selection process."""

    champion: CandidateResult | None
    passed_candidates: list[CandidateResult]
    rejected_candidates: list[CandidateResult]


class ModelSelectionEngine:
    """Filters candidate results against quality gates and selects champion."""

    def __init__(
        self,
        primary_metric: str = "macro_f1",
        direction: str = "maximize",
        gates: dict[str, float] | None = None,
    ) -> None:
        self.primary_metric = primary_metric
        self.direction = direction
        self.gates = gates or {}

    def select_champion(self, candidates: list[CandidateResult]) -> SelectionResult:
        """Filter candidates using quality gates and rank top performer."""
        logger.info("Selecting champion from %d candidate(s) with gates=%s", len(candidates), self.gates)
        passed: list[CandidateResult] = []
        rejected: list[CandidateResult] = []

        for candidate in candidates:
            if self._passes_gates(candidate.metrics):
                passed.append(candidate)
            else:
                rejected.append(candidate)
                logger.warning(
                    "Candidate %s rejected for %s: metrics=%s",
                    candidate.candidate_id,
                    candidate.model_name,
                    candidate.metrics,
                )

        if not passed:
            logger.warning("No candidate passed quality gates. Selection result: champion=None")
            return SelectionResult(
                champion=None, passed_candidates=[], rejected_candidates=rejected
            )

        reverse_sort = self.direction == "maximize"
        sorted_candidates = sorted(
            passed,
            key=lambda c: c.metrics.get(self.primary_metric, 0.0),
            reverse=reverse_sort,
        )

        logger.info(
            "Champion selected: %s with %s=%s",
            sorted_candidates[0].model_name,
            self.primary_metric,
            sorted_candidates[0].metrics.get(self.primary_metric, 0.0),
        )
        return SelectionResult(
            champion=sorted_candidates[0],
            passed_candidates=sorted_candidates,
            rejected_candidates=rejected,
        )

    def _passes_gates(self, metrics: dict[str, float]) -> bool:
        for metric, min_threshold in self.gates.items():
            if metrics.get(metric, 0.0) < min_threshold:
                return False
        return True