"""Tests for paired JSONL analysis summaries."""

from __future__ import annotations

import json

import pytest

from analysis.analyze import analyze_rows, load_jsonl


def _rows(config_hash: str = "cfg"):
    return [
        {"config_hash": config_hash, "sample_id": "a", "task_identity": "t1", "method": "rag", "status": "ok", "score": 0.2, "duration_s": 1, "generation_calls": 1, "generation_input_tokens": 10, "embedding_calls": 0, "context_length_tokens": 100},
        {"config_hash": config_hash, "sample_id": "a", "task_identity": "t1", "method": "rlm_depth1", "status": "ok", "score": 0.8, "duration_s": 2, "generation_calls": 2, "generation_input_tokens": 20, "embedding_calls": 0, "context_length_tokens": 100},
        {"config_hash": config_hash, "sample_id": "b", "task_identity": "t2", "method": "rag", "status": "ok", "score": 1.0, "duration_s": 1, "generation_calls": 1, "generation_input_tokens": 10, "embedding_calls": 0, "context_length_tokens": 200},
        {"config_hash": config_hash, "sample_id": "b", "task_identity": "t2", "method": "rlm_depth1", "status": "error", "score": 0.0, "duration_s": 2, "generation_calls": 1, "generation_input_tokens": 10, "embedding_calls": 0, "context_length_tokens": 200},
    ]


def test_analysis_reports_coverage_and_only_matched_paired_effects():
    summary = analyze_rows(_rows(), baseline="rag", bootstrap_iterations=100, seed=3)

    assert summary["methods"]["rag"]["coverage"] == 1.0
    assert summary["methods"]["rlm_depth1"]["coverage"] == 0.5
    comparison = summary["paired_comparisons"][0]
    assert comparison["n_pairs"] == 1
    assert comparison["mean_difference"] == pytest.approx(0.6)



def test_analysis_groups_paired_effects_by_declared_condition():
    rows = _rows()
    for row in rows:
        row["sample_metadata"] = {
            "num_records": 100 if row["sample_id"] == "a" else 200
        }

    summary = analyze_rows(
        rows,
        baseline="rag",
        condition_field="num_records",
        bootstrap_iterations=100,
    )

    comparison = summary["paired_by_condition"]["100"][0]
    assert comparison["n_pairs"] == 1
    assert comparison["mean_difference"] == pytest.approx(0.6)

def test_analysis_refuses_to_mix_configuration_hashes():
    with pytest.raises(ValueError, match="multiple configurations"):
        analyze_rows(_rows("a") + _rows("b"), bootstrap_iterations=10)


def test_jsonl_loader_ignores_only_a_truncated_final_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(_rows()[0]) + "\n{\"sample_id\":", encoding="utf-8")

    assert load_jsonl(path) == [_rows()[0]]
