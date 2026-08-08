"""Full-context baseline using the same subscription-backed model as RLM."""

from __future__ import annotations

from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.model_client import ModelClient, get_default_client
from src.trace.tracer import TraceNode

SYSTEM = """Answer from the provided document only.
Return JSON with keys `answer`, `confidence`, and `reasoning`.
Keep `answer` to the shortest fact or phrase that answers the question."""

PROMPT = """Document:
{document}

Question: {question}
"""


class ContextWindowExceeded(ValueError):
    """The complete prompt cannot fit without changing baseline semantics."""


class FullContextController(BaseController):
    """Send the complete document once; never silently truncate it."""

    method_name = "fullcontext"
    requires_retriever = False

    def __init__(
        self,
        model: str | None = None,
        *,
        client: ModelClient | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        del model
        self.client = client or get_default_client()
        self.max_input_tokens = max_input_tokens or settings.fullcontext_max_tokens

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        del kwargs
        trace = TraceNode(action="fullcontext", input=question)
        prompt = PROMPT.format(document=store.full_text, question=question)
        estimated_tokens = _estimate_tokens(prompt) + 100
        trace.metadata["estimated_input_tokens"] = estimated_tokens
        if estimated_tokens > self.max_input_tokens:
            trace.metadata["context_window_exceeded"] = True
            trace.finish()
            raise ContextWindowExceeded(
                f"Estimated input is {estimated_tokens} tokens; full-context limit is "
                f"{self.max_input_tokens}. Refusing to truncate the baseline."
            )

        payload = self.client.generate_json(prompt, system=SYSTEM, max_tokens=500)
        answer = str(payload.get("answer", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        trace.output = answer
        trace.metadata.update({"confidence": confidence, "context_window_exceeded": False})
        trace.finish()
        return ControllerResult(
            answer=answer,
            confidence=confidence,
            method=self.method_name,
            trace=trace,
            metadata={"reasoning": str(payload.get("reasoning", ""))},
        )


def _estimate_tokens(text: str) -> int:
    """Conservative provider-independent approximation used only as a guard."""
    return max(1, (len(text) + 3) // 4)
