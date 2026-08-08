"""Tests for trace data structures."""

import json
import tempfile
from pathlib import Path

import pytest
from src.trace.tracer import TraceNode
from src.trace.trace_viewer import export_trace, pretty_print


def test_trace_node_basic():
    node = TraceNode(action="test", input="in", output="out")
    assert node.action == "test"
    assert node.input == "in"
    assert node.output == "out"


def test_trace_tree():
    root = TraceNode(action="root")
    child1 = root.add_child(TraceNode(action="child1"))
    child2 = root.add_child(TraceNode(action="child2"))
    child1.add_child(TraceNode(action="grandchild"))

    assert len(root.children) == 2
    assert root.max_depth() == 2
    assert root.node_count() == 4


def test_to_dict():
    root = TraceNode(action="root", input="q", output="a", metadata={"conf": 0.9})
    root.add_child(TraceNode(action="child"))
    d = root.to_dict()

    assert d["action"] == "root"
    assert len(d["children"]) == 1



def test_to_dict_preserves_complete_repl_code_for_audit():
    code = "answer = " + repr("x" * 500)
    node = TraceNode(action="repl_exec", input=code)

    assert node.to_dict()["input"] == code

def test_export_trace():
    root = TraceNode(action="test", output="result")
    root.add_child(TraceNode(action="step1"))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.json"
        export_trace(root, path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["action"] == "test"


def test_pretty_print():
    root = TraceNode(action="rlm", input="question")
    root.add_child(TraceNode(action="search", output="[0, 1, 2]"))
    text = pretty_print(root)

    assert "[rlm]" in text
    assert "[search]" in text
