"""Paper-faithful Recursive Language Model controller.

The complete document lives as ``context`` inside a persistent Python REPL.
The root model sees only metadata and REPL observations; it can inspect the
context symbolically and invoke ordinary sub-model calls from generated code.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.controllers.base import BaseController, ControllerResult
from src.environment.document_store import DocumentStore
from src.model_client import ModelClient, get_default_client
from src.trace.tracer import TraceNode

logger = logging.getLogger(__name__)

ROOT_SYSTEM = """You are a Recursive Language Model operating a persistent Python REPL.

The full source document is stored in the variable `context`; it is NOT present in this prompt.
Write Python that inspects, slices, searches, transforms, and aggregates `context`.
Persistent variables survive across iterations.

Available functions:
- llm_query(prompt): one ordinary language-model call over a programmatically constructed prompt.
- llm_query_batched(prompts): sequential ordinary calls for several prompts.
- rlm_query(subcontext, question): a recursive RLM call when depth permits.
- FINAL_VAR(name): return the value of a REPL variable as the final answer.

Return JSON only: {"thought": "brief plan", "code": "Python code"}.
Prefer symbolic inspection before sub-calls. Batch related records to conserve subscription usage.
Never invent evidence that was not read from `context` or returned by a sub-call.
"""

ROOT_PROMPT = """Question: {question}
REPL depth: {depth}/{max_depth}
Iteration: {iteration}/{max_iterations}
Context metadata: {context_chars} characters, {context_lines} lines
Persistent variables:
{namespace}

Recent REPL history:
{history}

Last observation:
{observation}

Write the next Python code to execute. Set a variable and call FINAL_VAR("variable_name") when done.
"""

SUBCALL_SYSTEM = """Answer the request using only the supplied content. Return only the requested result; do not add a preamble."""

SYNTHESIS_SYSTEM = """Produce the shortest answer supported by the REPL observations. Return only the answer."""

SYNTHESIS_PROMPT = """Question: {question}
Persistent variables:
{namespace}
Recent observations:
{history}
Return the best supported final answer."""


@dataclass
class _CallBudget:
    limit: int
    used: int = 0
    max_depth_reached: int = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError(f"Sub-call budget exhausted ({self.limit}).")
        self.used += 1


@dataclass
class _Outcome:
    answer: str
    confidence: float


class RLMController(BaseController):
    """Persistent prompt-as-variable REPL with bounded symbolic sub-calls."""

    method_name = "rlm"
    requires_retriever = False

    def __init__(
        self,
        model: str | None = None,
        *,
        client: ModelClient | None = None,
        max_depth: int | None = None,
        max_iterations: int | None = None,
        max_subcalls: int | None = None,
        allow_subcalls: bool = True,
    ) -> None:
        del model  # Model selection belongs to the injected client.
        self.client = client or get_default_client()
        self.max_depth = settings.rlm_max_depth if max_depth is None else max_depth
        self.max_iterations = (
            settings.rlm_max_iterations if max_iterations is None else max_iterations
        )
        self.max_subcalls = settings.rlm_max_subcalls if max_subcalls is None else max_subcalls
        self.allow_subcalls = allow_subcalls
        self.history_chars = settings.rlm_history_chars
        self.stdout_chars = settings.rlm_stdout_chars

    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        del kwargs
        trace = TraceNode(
            action="rlm",
            input=question,
            metadata={
                "mode": "paper_repl",
                "max_depth": self.max_depth,
                "allow_subcalls": self.allow_subcalls,
            },
        )
        budget = _CallBudget(limit=self.max_subcalls)
        outcome = self._run_repl(
            question=question,
            context=store.full_text,
            depth=0,
            budget=budget,
            trace=trace,
        )
        trace.output = outcome.answer
        trace.metadata.update(
            {
                "confidence": outcome.confidence,
                "subcalls": budget.used,
                "rlm_depth_reached": budget.max_depth_reached,
                "allow_subcalls": self.allow_subcalls,
            }
        )
        trace.finish()
        return ControllerResult(
            answer=outcome.answer,
            confidence=outcome.confidence,
            method=self.method_name,
            trace=trace,
            metadata={
                "controller_style": "paper_repl",
                "subcalls": budget.used,
                "rlm_depth_reached": budget.max_depth_reached,
                "allow_subcalls": self.allow_subcalls,
            },
        )

    def _run_repl(
        self,
        *,
        question: str,
        context: str,
        depth: int,
        budget: _CallBudget,
        trace: TraceNode,
    ) -> _Outcome:
        budget.max_depth_reached = max(budget.max_depth_reached, depth)
        namespace: dict[str, Any] = {"context": context, "notes": ""}
        history: list[dict[str, str]] = []
        state: dict[str, Any] = {"final": None}
        observation = "REPL initialized. Inspect the context variable."

        for iteration in range(1, self.max_iterations + 1):
            prompt = ROOT_PROMPT.format(
                question=question,
                depth=depth,
                max_depth=self.max_depth,
                iteration=iteration,
                max_iterations=self.max_iterations,
                context_chars=len(context),
                context_lines=context.count("\n") + 1,
                namespace=self._summarize_namespace(namespace),
                history=self._format_history(history),
                observation=self._clip(observation, self.stdout_chars),
            )
            system = ROOT_SYSTEM
            if not self.allow_subcalls:
                system += (
                    "\nModel subcalls are disabled for this ablation. Solve only "
                    "with Python operations over `context`."
                )
            decision = self.client.generate_json(
                prompt,
                system=system,
                max_tokens=1200,
            )
            thought = str(decision.get("thought", "")).strip()
            code = self._strip_code_fence(str(decision.get("code", "")))
            plan = trace.add_child(
                TraceNode(
                    action="plan",
                    input=question,
                    output=thought,
                    metadata={"depth": depth, "iteration": iteration},
                )
            )
            execution = plan.add_child(
                TraceNode(
                    action="repl_exec",
                    input=code,
                    metadata={"depth": depth, "iteration": iteration},
                )
            )
            observation = self._execute(
                code=code,
                question=question,
                namespace=namespace,
                state=state,
                depth=depth,
                budget=budget,
                trace=execution,
            )
            execution.output = self._clip(observation, self.stdout_chars)
            execution.finish()
            plan.finish()
            history.append({"thought": thought, "code": code, "observation": observation})
            if state["final"] is not None:
                return _Outcome(answer=str(state["final"]).strip(), confidence=1.0)

        synthesis = trace.add_child(
            TraceNode(action="synthesize", input=question, metadata={"depth": depth})
        )
        result = self.client.generate(
            SYNTHESIS_PROMPT.format(
                question=question,
                namespace=self._summarize_namespace(namespace),
                history=self._format_history(history),
            ),
            system=SYNTHESIS_SYSTEM,
            max_tokens=300,
        )
        synthesis.output = result.text
        synthesis.finish()
        return _Outcome(answer=result.text.strip(), confidence=0.5)

    def _execute(
        self,
        *,
        code: str,
        question: str,
        namespace: dict[str, Any],
        state: dict[str, Any],
        depth: int,
        budget: _CallBudget,
        trace: TraceNode,
    ) -> str:
        if not code:
            return "Planner returned no code."
        local_values = dict(namespace)


        def llm_query(prompt: str, model: str | None = None) -> str:
            if not self.allow_subcalls:
                raise RuntimeError("Model subcalls are disabled for this ablation.")
            del model
            budget.consume()
            node = trace.add_child(
                TraceNode(
                    action="llm_query",
                    input=self._clip(str(prompt), 200),
                    metadata={"depth": depth, "subcall": budget.used},
                )
            )
            result = self.client.generate(str(prompt), system=SUBCALL_SYSTEM, max_tokens=1000)
            node.output = self._clip(result.text, 500)
            node.metadata.update(
                {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                }
            )
            node.finish()
            return result.text

        def llm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
            return [llm_query(prompt, model=model) for prompt in prompts]

        def rlm_query(subcontext: str, sub_question: str) -> str:
            if not self.allow_subcalls:
                raise RuntimeError("Model subcalls are disabled for this ablation.")
            if depth >= self.max_depth:
                return llm_query(f"Context:\n{subcontext}\n\nQuestion: {sub_question}")
            budget.consume()
            node = trace.add_child(
                TraceNode(
                    action="rlm_query",
                    input=sub_question,
                    metadata={"depth": depth + 1, "subcall": budget.used},
                )
            )
            result = self._run_repl(
                question=str(sub_question),
                context=str(subcontext),
                depth=depth + 1,
                budget=budget,
                trace=node,
            )
            node.output = self._clip(result.answer, 500)
            node.finish()
            return result.answer

        def final_var(name: Any) -> None:
            if isinstance(name, str) and name in local_values:
                state["final"] = local_values[name]
            else:
                state["final"] = name
            trace.add_child(
                TraceNode(
                    action="final_var",
                    input=str(name),
                    output=self._clip(str(state["final"]), 500),
                    metadata={"depth": depth},
                )
            ).finish()

        tools: dict[str, Any] = {
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
            "rlm_query": rlm_query,
            "FINAL_VAR": final_var,
        }
        local_values.update(tools)
        local_values["__builtins__"] = self._safe_builtins()
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(code, local_values, local_values)
        except Exception as exc:
            logger.debug("RLM execution error: %s", exc)
            return f"Execution error: {type(exc).__name__}: {exc}"

        namespace.clear()
        for key, value in local_values.items():
            if key in tools or key.startswith("__"):
                continue
            namespace[key] = value
        printed = stdout.getvalue().strip()
        summary = self._summarize_namespace(namespace)
        return f"Printed:\n{printed}\n\nVariables:\n{summary}" if printed else f"Variables:\n{summary}"

    def _summarize_namespace(self, namespace: dict[str, Any]) -> str:
        values: list[str] = []
        for key in sorted(namespace):
            if key == "context" or key.startswith("_"):
                continue
            value = namespace[key]
            if callable(value):
                continue
            values.append(f"{key} = {self._summarize_value(value)}")
        return "\n".join(values) if values else "(no persistent variables)"

    def _summarize_value(self, value: Any) -> str:
        if isinstance(value, str):
            return repr(self._clip(value, 600))
        if isinstance(value, list):
            preview = ", ".join(self._summarize_value(item) for item in value[:5])
            suffix = "" if len(value) <= 5 else f", ... ({len(value)} items)"
            return f"[{preview}{suffix}]"
        if isinstance(value, dict):
            items = list(value.items())[:5]
            preview = ", ".join(f"{key!r}: {self._summarize_value(item)}" for key, item in items)
            suffix = "" if len(value) <= 5 else ", ..."
            return "{" + preview + suffix + "}"
        return self._clip(repr(value), 600)

    def _format_history(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "(none)"
        rendered = []
        for index, item in enumerate(history[-6:], start=max(1, len(history) - 5)):
            rendered.append(
                f"Iteration {index}\nThought: {self._clip(item['thought'], 300)}\n"
                f"Code: {self._clip(item['code'], 800)}\n"
                f"Observation: {self._clip(item['observation'], 1200)}"
            )
        return self._clip("\n\n".join(rendered), self.history_chars)

    def _strip_code_fence(self, code: str) -> str:
        text = code.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines.pop()
            text = "\n".join(lines).strip()
        return text

    def _clip(self, value: str, limit: int) -> str:
        return value if len(value) <= limit else value[: limit - 3] + "..."

    def _safe_builtins(self) -> dict[str, Any]:
        return {
            "abs": abs,
            "all": all,
            "any": any,
            "BaseException": BaseException,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "Exception": Exception,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "type": type,
            "ValueError": ValueError,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
