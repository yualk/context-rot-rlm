"""Provider-neutral text generation through the local subscription harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from src.cost_tracker import CostTracker, tracker


@dataclass(frozen=True)
class GenerationUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    billed_cost_usd: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    usage: GenerationUsage


class ModelClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> GenerationResult: ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


class SubscriptionLimitError(RuntimeError):
    """Raised before a subscription-backed client exceeds its local call cap."""


class OmpSubscriptionClient:
    """Invoke an authenticated OMP model without using API-key billing.

    Each request is an isolated, tool-free OMP print session. The input is passed
    through a temporary file so long prompts do not hit Windows command-line
    limits. ``call_limit`` is process-local and intentionally conservative.
    """

    def __init__(
        self,
        *,
        executable: str = "omp",
        model: str = "openai-codex/gpt-5.6-luna",
        thinking: str = "low",
        timeout_s: int = 300,
        call_limit: int = 40,
        temp_dir: str | Path | None = None,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        usage_tracker: CostTracker | None = tracker,
    ) -> None:
        self.executable = executable
        self.model = model
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.call_limit = call_limit
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.run_command = run_command
        self.usage_tracker = usage_tracker
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> GenerationResult:
        del max_tokens  # OMP model output limits are controlled by the selected model.
        if self.calls >= self.call_limit:
            raise SubscriptionLimitError(
                f"Subscription call cap reached ({self.call_limit}); refusing another request."
            )

        self.calls += 1
        prompt_path = self._write_prompt(prompt)
        try:
            args = [
                self.executable,
                "-p",
                "--model",
                self.model,
                "--thinking",
                self.thinking,
                "--no-tools",
                "--no-session",
                "--no-skills",
                "--no-rules",
                "--no-extensions",
                "--mode",
                "json",
                "--max-time",
                f"{self.timeout_s}s",
                "--system-prompt",
                system or "Follow the user request precisely.",
                f"@{prompt_path}",
            ]
            completed = self.run_command(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout_s + 30,
                check=False,
            )
        finally:
            prompt_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(f"OMP subscription request failed: {detail}")

        result = self._parse_output(completed.stdout)
        if self.usage_tracker is not None:
            self.usage_tracker.record_generation(
                provider=result.usage.provider,
                model=result.usage.model,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cache_read_tokens=result.usage.cache_read_tokens,
                cache_write_tokens=result.usage.cache_write_tokens,
                cost_usd=result.usage.billed_cost_usd,
                estimated_cost_usd=result.usage.estimated_cost_usd,
            )
        return result

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        result = self.generate(prompt, system=system, max_tokens=max_tokens)
        return _parse_json_object(result.text)

    def _write_prompt(self, prompt: str) -> Path:
        if self.temp_dir is not None:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix="rlm_prompt_",
            dir=self.temp_dir,
            delete=False,
        ) as handle:
            handle.write(prompt)
            return Path(handle.name)

    def _parse_output(self, output: str) -> GenerationResult:
        candidates: list[dict[str, Any]] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message_end":
                continue
            message = event.get("message", {})
            if message.get("role") == "assistant":
                candidates.append(message)

        if not candidates:
            raise RuntimeError("OMP returned no assistant message.")

        message = next(
            (item for item in reversed(candidates) if item.get("usage", {}).get("input", 0)),
            candidates[-1],
        )
        text = "".join(
            str(part.get("text", ""))
            for part in message.get("content", [])
            if part.get("type") == "text"
        )
        usage = message.get("usage", {})
        costs = usage.get("cost", {})
        model_id = str(message.get("model", self.model.rsplit("/", 1)[-1]))
        return GenerationResult(
            text=text.strip(),
            usage=GenerationUsage(
                provider=str(message.get("provider", "openai-codex")),
                model=model_id,
                input_tokens=int(usage.get("input", 0) or 0),
                output_tokens=int(usage.get("output", 0) or 0),
                cache_read_tokens=int(usage.get("cacheRead", 0) or 0),
                cache_write_tokens=int(usage.get("cacheWrite", 0) or 0),
                estimated_cost_usd=float(costs.get("total", 0.0) or 0.0),
                billed_cost_usd=0.0,
            ),
        )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model.")
    return value


def get_default_client() -> OmpSubscriptionClient:
    from src.config import settings

    return OmpSubscriptionClient(
        executable=settings.omp_executable,
        model=settings.model_generation,
        thinking=settings.model_thinking,
        timeout_s=settings.model_timeout_s,
        call_limit=settings.subscription_call_limit,
    )
