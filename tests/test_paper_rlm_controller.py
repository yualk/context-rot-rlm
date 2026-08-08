"""Observable contracts for the paper-faithful RLM controller."""

from __future__ import annotations

from collections import deque
import json


from src.controllers.rlm_controller import RLMController
from src.environment.document_store import DocumentStore
from src.model_client import GenerationResult, GenerationUsage


class ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = deque(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 4096):
        self.prompts.append(prompt)
        return GenerationResult(
            text=self.responses.popleft(),
            usage=GenerationUsage(provider="test", model="scripted"),
        )

    def generate_json(self, prompt: str, *, system: str = "", max_tokens: int = 4096):
        import json

        return json.loads(self.generate(prompt, system=system, max_tokens=max_tokens).text)


def _store(text: str) -> DocumentStore:
    store = DocumentStore()
    store.ingest(text, doc_id="fixture")
    return store


def test_rlm_exposes_complete_document_as_context_variable():
    client = ScriptedClient([
        json.dumps({
            "thought": "inspect context",
            "code": "\n".join([
                'kind = type(context).__name__',
                'line = [x for x in context.splitlines() if x.startswith("needle:")][0]',
                'answer = line.split(":", 1)[1].strip()',
                'FINAL_VAR("answer")',
            ]),
        })
    ])
    controller = RLMController(client=client, max_depth=1, max_iterations=2)

    result = controller.answer(
        "What is the needle?",
        _store("irrelevant\nneedle: cobalt-42\nmore irrelevant"),
    )

    assert result.answer == "cobalt-42"
    assert result.metadata["controller_style"] == "paper_repl"
    assert result.metadata["subcalls"] == 0
    assert result.trace is not None
    assert not any(node.action in {"bootstrap", "search"} for node in result.trace.walk())


def test_rlm_can_programmatically_invoke_single_sub_model_calls():
    client = ScriptedClient([
        json.dumps({
            "thought": "delegate slice",
            "code": "\n".join([
                'answer = llm_query("Return only beta")',
                'FINAL_VAR("answer")',
            ]),
        }),
        "beta",
    ])
    controller = RLMController(client=client, max_depth=1, max_iterations=2, max_subcalls=2)

    result = controller.answer("Find the answer", _store("alpha beta gamma"))

    assert result.answer == "beta"
    assert result.metadata["subcalls"] == 1
    assert any(node.action == "llm_query" for node in result.trace.walk())
