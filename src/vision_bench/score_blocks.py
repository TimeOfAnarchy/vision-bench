from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thresholds import Thresholds


@dataclass(frozen=True)
class MappedScores:
    native: np.ndarray
    groups: list[list[int]]
    native_labels: list[str]
    cutoffs: dict[int, float]

    def block(
        self, rows: np.ndarray, columns: list[int], *, with_decisions: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        groups = [self.groups[column] for column in columns]
        first = [group[0] for group in groups]
        scores = self.native[np.ix_(rows, first)]
        decisions = (
            scores >= np.asarray([self.cutoffs[i] for i in first], dtype=np.float32)
            if with_decisions
            else None
        )
        for column, group in enumerate(groups):
            for index in group[1:]:
                extra = self.native[rows, index]
                scores[:, column] = np.maximum(scores[:, column], extra)
                if decisions is not None:
                    decisions[:, column] |= extra >= np.float32(self.cutoffs[index])
        return scores, decisions


def score_block(scores: np.ndarray | MappedScores, rows: np.ndarray, columns: list[int]) -> np.ndarray:
    if isinstance(scores, MappedScores):
        return scores.block(rows, columns)[0]
    return scores[np.ix_(rows, columns)]


def scored_decisions(
    scores: np.ndarray | MappedScores,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    thresholds: float | Thresholds,
    *,
    dtype=None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(scores, MappedScores):
        block, decisions = scores.block(rows, columns, with_decisions=True)
        assert decisions is not None
        return block, decisions
    block = np.asarray(score_block(scores, rows, columns), dtype=dtype)
    cutoff = thresholds.for_labels(labels) if isinstance(thresholds, Thresholds) else thresholds
    return block, block >= cutoff


def describe_thresholds(scores: np.ndarray | MappedScores, labels: list[str], thresholds: Thresholds) -> dict:
    return thresholds.describe(scores.native_labels if isinstance(scores, MappedScores) else labels)
