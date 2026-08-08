"""Observable contracts for comparable full-context and BM25 baselines."""

from __future__ import annotations

import json

import pytest

from src.controllers.fullcontext import ContextWindowExceeded, FullContextController
from src.controllers.rag_baseline import RAGController
from src.environment.document_store import DocumentStore
from tests.test_paper_rlm_controller import ScriptedClient


def _store(text: str) -> DocumentStore:
    store = DocumentStore(chunk_size=4, chunk_overlap=1, min_chunk_size=1)
    store.ingest(text, doc_id="fixture")
    return store


def test_fullcontext_sends_complete_document_without_truncation():
    client = ScriptedClient([
        json.dumps({"answer": "cobalt", "confidence": 1.0, "reasoning": "present"})
    ])
    controller = FullContextController(client=client, max_input_tokens=1_000)

    result = controller.answer("Which color?", _store("first sentinel cobalt last"))

    assert result.answer == "cobalt"
    assert "first sentinel cobalt last" in client.prompts[0]


def test_fullcontext_reports_context_limit_instead_of_silent_truncation():
    controller = FullContextController(client=ScriptedClient([]), max_input_tokens=1)

    with pytest.raises(ContextWindowExceeded, match="Refusing to truncate"):
        controller.answer("Which color?", _store("first sentinel cobalt last"))


def test_rag_uses_bm25_and_reports_retrieval_work_separately():
    client = ScriptedClient([
        json.dumps({"answer": "cobalt", "confidence": 0.9, "reasoning": "retrieved"})
    ])
    controller = RAGController(client=client, top_k=1)

    result = controller.answer(
        "What follows uniquequery?",
        _store("irrelevant words here now uniquequery cobalt answer here"),
    )

    assert result.answer == "cobalt"
    assert "uniquequery cobalt" in client.prompts[0]
    assert result.metadata["retrieval_build_s"] >= 0
    assert result.metadata["retrieval_query_s"] >= 0
    assert result.trace.metadata["retriever"] == "bm25"
