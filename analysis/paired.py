"""Paired method comparisons with task-clustered bootstrap intervals."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable


@dataclass(frozen=True)
class PairedDifference:
    baseline: str
    treatment: str
    metric: str
    n_pairs: int
    n_clusters: int
    mean_difference: float
    ci_low: float | None
    ci_high: float | None
    wins: int
    ties: int
    losses: int


def paired_method_difference(
    rows: Iterable[dict[str, Any]],
    *,
    baseline: str,
    treatment: str,
    metric: str = "score",
    cluster_field: str = "task_identity",
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedDifference:
    """Compare methods on identical successful samples.

    Bootstrap resampling happens at the task-identity level, so repeated context
    lengths for one generated task remain in the same resampled cluster.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("method") not in {baseline, treatment}:
            continue
        sample_id = str(row["sample_id"])
        indexed[(sample_id, str(row["method"]))] = row

    clustered: dict[str, list[float]] = {}
    for sample_id, method in list(indexed):
        if method != baseline:
            continue
        base_row = indexed[(sample_id, baseline)]
        treatment_row = indexed.get((sample_id, treatment))
        if treatment_row is None or metric not in base_row or metric not in treatment_row:
            continue
        difference = float(treatment_row[metric]) - float(base_row[metric])
        cluster = str(
            treatment_row.get(cluster_field)
            or base_row.get(cluster_field)
            or sample_id
        )
        clustered.setdefault(cluster, []).append(difference)

    differences = [value for cluster in clustered.values() for value in cluster]
    if not differences:
        raise ValueError(f"No successful paired rows for {baseline!r} and {treatment!r}.")

    cluster_names = sorted(clustered)
    ci_low: float | None = None
    ci_high: float | None = None
    if len(cluster_names) >= 2:
        rng = random.Random(seed)
        bootstrapped: list[float] = []
        for _ in range(iterations):
            selected = [rng.choice(cluster_names) for _ in cluster_names]
            draw = [value for name in selected for value in clustered[name]]
            bootstrapped.append(fmean(draw))
        bootstrapped.sort()
        alpha = 1.0 - confidence
        lower_index = max(0, int((alpha / 2.0) * (iterations - 1)))
        upper_index = min(iterations - 1, int((1.0 - alpha / 2.0) * (iterations - 1)))
        ci_low = bootstrapped[lower_index]
        ci_high = bootstrapped[upper_index]

    tolerance = 1e-12
    return PairedDifference(
        baseline=baseline,
        treatment=treatment,
        metric=metric,
        n_pairs=len(differences),
        n_clusters=len(clustered),
        mean_difference=fmean(differences),
        ci_low=ci_low,
        ci_high=ci_high,
        wins=sum(value > tolerance for value in differences),
        ties=sum(abs(value) <= tolerance for value in differences),
        losses=sum(value < -tolerance for value in differences),
    )
