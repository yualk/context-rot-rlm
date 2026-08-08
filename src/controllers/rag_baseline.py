"""Single-pass BM25 retrieval baseline."""

from __future__ import annotations

import time
from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.bm25_index import BM25Index
from src.environment.document_store import DocumentStore
from src.model_client import ModelClient, get_default_client
from src.trace.tracer import TraceNode

SYSTEM = """Answer the question using only the retrieved evidence.
Return JSON with keys `answer`, `confidence`, and `reasoning`.
Keep `answer` short. If the evidence is insufficient, use an empty answer and confidence 0."""

PROMPT = """Evidence:
{evidence}

Question: {question}
"""


class RAGController(BaseController):
    """Build a BM25 index, retrieve once, and generate once."""

    method_name = "rag"
    requires_retriever = False

    def __init__(
        self,
        model: str | None = None,
        top_k: int | None = None,
        *,
        client: ModelClient | None = None,
    ) -> None:
        del model
        self.client = client or get_default_client()
        self.top_k = top_k or settings.rag_top_k

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        del kwargs
        trace = TraceNode(action="rag", input=question, metadata={"retriever": "bm25"})

        build_start = time.perf_counter()
        index = BM25Index(top_k=self.top_k)
        index.build(store.chunks)
        build_s = time.perf_counter() - build_start

        query_start = time.perf_counter()
        hits = index.search(question, top_k=self.top_k)
        query_s = time.perf_counter() - query_start
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        evidence = store.get_chunks_text(chunk_ids)

        search_node = trace.add_child(
            TraceNode(
                action="search",
                input=question,
                output=str(chunk_ids),
                metadata={
                    "retriever": "bm25",
                    "top_k": self.top_k,
                    "build_s": build_s,
                    "query_s": query_s,
                    "hit_count": len(chunk_ids),
                },
            )
        )
        search_node.finish()

        payload = self.client.generate_json(
            PROMPT.format(evidence=evidence, question=question),
            system=SYSTEM,
            max_tokens=500,
        )
        answer = str(payload.get("answer", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        reason_node = trace.add_child(
            TraceNode(
                action="reason",
                input=question,
                output=answer,
                metadata={"confidence": confidence},
            )
        )
        reason_node.finish()
        trace.output = answer
        trace.metadata.update(
            {
                "confidence": confidence,
                "retrieval_build_s": build_s,
                "retrieval_query_s": query_s,
                "retrieved_chunks": len(chunk_ids),
            }
        )
        trace.finish()
        return ControllerResult(
            answer=answer,
            confidence=confidence,
            method=self.method_name,
            trace=trace,
            metadata={
                "reasoning": str(payload.get("reasoning", "")),
                "retrieval_build_s": build_s,
                "retrieval_query_s": query_s,
                "retrieved_chunks": len(chunk_ids),
            },
        )
