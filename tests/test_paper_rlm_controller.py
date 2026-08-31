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


def test_rlm_can_disable_all_model_subcalls():
    client = ScriptedClient([
        json.dumps({
            "thought": "try a prohibited subcall",
            "code": 'answer = llm_query("Return beta")',
        }),
        json.dumps({
            "thought": "solve from context instead",
            "code": "\n".join([
                'answer = context.split()[-1]',
                'FINAL_VAR("answer")',
            ]),
        }),
    ])
    controller = RLMController(
        client=client,
        max_depth=0,
        max_iterations=2,
        allow_subcalls=False,
    )

    result = controller.answer("Find the answer", _store("alpha beta"))

    assert result.answer == "beta"
    assert result.metadata["subcalls"] == 0
    assert result.metadata["allow_subcalls"] is False
    assert not any(node.action in {"llm_query", "rlm_query"} for node in result.trace.walk())


def test_depth_one_rlm_recurses_over_a_programmatic_subcontext():
    client = ScriptedClient([
        json.dumps({
            "thought": "delegate the section recursively",
            "code": "\n".join([
                'answer = rlm_query(context, "Return the final word")',
                'FINAL_VAR("answer")',
            ]),
        }),
        json.dumps({
            "thought": "solve the delegated section",
            "code": "\n".join([
                'answer = context.split()[-1]',
                'FINAL_VAR("answer")',
            ]),
        }),
    ])
    controller = RLMController(
        client=client,
        max_depth=1,
        max_iterations=2,
        max_subcalls=2,
    )

    result = controller.answer("Find the answer", _store("alpha beta"))

    assert result.answer == "beta"
    assert result.metadata["subcalls"] == 1
    assert result.metadata["rlm_depth_reached"] == 1
    assert any(node.action == "rlm_query" for node in result.trace.walk())


def test_rlm_allows_safe_standard_library_imports():
    client = ScriptedClient([
        json.dumps({
            "thought": "extract with a regular expression",
            "code": "\n".join([
                "import re",
                "answer = re.search(r'needle: (\\S+)', context).group(1)",
                'FINAL_VAR("answer")',
            ]),
        }),
    ])
    controller = RLMController(client=client, max_iterations=1)

    result = controller.answer("Find the needle", _store("needle: cobalt-42"))

    assert result.answer == "cobalt-42"
