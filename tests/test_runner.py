"""Observable contracts for the paired, resumable experiment runner."""

from __future__ import annotations

from benchmarks.diagnostic import DiagnosticSample
from experiments.runner import ExperimentRunner
from src.controllers.base import BaseController, ControllerResult
from src.trace.tracer import TraceNode


class FixedController(BaseController):
    requires_retriever = False

    def __init__(self, method: str, calls: list[tuple[str, str]]) -> None:
        self.method_name = method
        self.calls = calls

    def answer(self, question, store, **kwargs):
        self.calls.append((self.method_name, store.doc_id))
        trace = TraceNode(action=self.method_name, input=question, output="42")
        trace.finish()
        return ControllerResult(
            answer="42",
            confidence=1.0,
            method=self.method_name,
            trace=trace,
            metadata={"observed_chars": len(store.full_text)},
        )


def _sample(sample_id: str = "sample-1") -> DiagnosticSample:
    return DiagnosticSample(
        sample_id=sample_id,
        task_identity="task-1",
        document="record: answer 42",
        question="What is the answer?",
        answer="42",
        context_length_tokens=5,
        metadata={"benchmark": "sniah"},
    )


def test_runner_persists_each_paired_method_cell(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.runner._source_revision", lambda: "test-revision")
    calls = []
    runner = ExperimentRunner(
        samples=[_sample()],
        methods=["left", "right"],
        output_path=tmp_path / "results.jsonl",
        benchmark="fixture",
        controller_factory=lambda method: FixedController(method, calls),
    )

    rows = runner.run()

    assert {(row["sample_id"], row["method"]) for row in rows} == {
        ("sample-1", "left"),
        ("sample-1", "right"),
    }
    assert all(row["score"] == 1.0 for row in rows)
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_runner_resumes_at_sample_method_granularity(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.runner._source_revision", lambda: "test-revision")
    path = tmp_path / "results.jsonl"
    initial_calls = []
    ExperimentRunner(
        samples=[_sample()],
        methods=["left", "right"],
        output_path=path,
        benchmark="fixture",
        controller_factory=lambda method: FixedController(method, initial_calls),
    ).run()

    resumed_calls = []
    rows = ExperimentRunner(
        samples=[_sample()],
        methods=["left", "right"],
        output_path=path,
        benchmark="fixture",
        controller_factory=lambda method: FixedController(method, resumed_calls),
    ).run()

    assert resumed_calls == []
    assert len(rows) == 2


def test_runner_records_context_limit_without_marking_cell_complete(tmp_path, monkeypatch):
    from src.controllers.fullcontext import FullContextController
    from tests.test_paper_rlm_controller import ScriptedClient

    monkeypatch.setattr("experiments.runner._source_revision", lambda: "test-revision")
    runner = ExperimentRunner(
        samples=[_sample()],
        methods=["fullcontext"],
        output_path=tmp_path / "results.jsonl",
        benchmark="fixture",
        controller_factory=lambda method: FullContextController(
            client=ScriptedClient([]), max_input_tokens=1
        ),
    )

    rows = runner.run()

    assert rows[0]["status"] == "context_window_exceeded"
    assert runner.store.pending("fullcontext", ["sample-1"]) == ["sample-1"]
