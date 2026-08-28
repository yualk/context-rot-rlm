"""Paired, task-clustered statistical comparison contracts."""

import pytest

from analysis.paired import paired_method_difference


def test_paired_difference_uses_only_matching_successful_samples():
    rows = [
        {"sample_id": "a", "task_identity": "t1", "method": "rag", "status": "ok", "score": 0.2},
        {"sample_id": "a", "task_identity": "t1", "method": "rlm", "status": "ok", "score": 0.8},
        {"sample_id": "b", "task_identity": "t2", "method": "rag", "status": "ok", "score": 0.9},
        {"sample_id": "b", "task_identity": "t2", "method": "rlm", "status": "error", "score": 0.0},
        {"sample_id": "c", "task_identity": "t3", "method": "rlm", "status": "ok", "score": 1.0},
    ]

    result = paired_method_difference(rows, baseline="rag", treatment="rlm", iterations=200, seed=7)

    assert result.n_pairs == 1
    assert result.mean_difference == pytest.approx(0.6)
    assert result.wins == 1
    assert result.losses == 0
    assert result.ci_low is None
    assert result.ci_high is None


def test_clustered_bootstrap_is_deterministic_and_reports_interval():
    rows = []
    for task, values in {"t1": [(0.1, 0.5), (0.2, 0.4)], "t2": [(0.8, 0.7)]}.items():
        for index, (base, treatment) in enumerate(values):
            sample_id = f"{task}-{index}"
            rows.extend([
                {"sample_id": sample_id, "task_identity": task, "method": "rag", "status": "ok", "score": base},
                {"sample_id": sample_id, "task_identity": task, "method": "rlm", "status": "ok", "score": treatment},
            ])

    first = paired_method_difference(rows, baseline="rag", treatment="rlm", iterations=500, seed=19)
    second = paired_method_difference(rows, baseline="rag", treatment="rlm", iterations=500, seed=19)

    assert first == second
    assert first.ci_low <= first.mean_difference <= first.ci_high
    assert first.n_clusters == 2
