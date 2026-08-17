"""Tests for crash-safe, configuration-aware result persistence."""

from __future__ import annotations

import json

from experiments.result_store import ResultStore


def test_result_store_resumes_individual_sample_method_pairs(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultStore(path, config_hash="cfg-a")
    store.append({"sample_id": "s1", "method": "rlm", "status": "ok"})

    assert store.pending("rlm", ["s1", "s2"]) == ["s2"]
    assert store.pending("rag", ["s1", "s2"]) == ["s1", "s2"]


def test_result_store_does_not_reuse_rows_from_another_configuration(tmp_path):
    path = tmp_path / "results.jsonl"
    ResultStore(path, config_hash="cfg-a").append(
        {"sample_id": "s1", "method": "rlm", "status": "ok"}
    )

    store = ResultStore(path, config_hash="cfg-b")

    assert store.pending("rlm", ["s1"]) == ["s1"]


def test_result_store_writes_parseable_jsonl_after_each_sample(tmp_path):
    path = tmp_path / "results.jsonl"
    store = ResultStore(path, config_hash="cfg-a")
    store.append({"sample_id": "s1", "method": "rlm", "status": "ok"})
    store.append({"sample_id": "s2", "method": "rlm", "status": "error"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["s1", "s2"]
    assert all(row["config_hash"] == "cfg-a" for row in rows)


def test_result_store_writes_a_configuration_specific_manifest(tmp_path):
    path = tmp_path / "results.jsonl"
    manifest = {"model": "openai-codex/gpt-5.6-luna", "seed": 42}

    store = ResultStore(path, config_hash="cfg-a", manifest=manifest)

    saved = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert saved["config_hash"] == "cfg-a"
    assert saved["configuration"] == manifest
