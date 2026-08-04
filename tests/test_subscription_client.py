"""Behavioral tests for the subscription-backed model client."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from src.model_client import OmpSubscriptionClient


def test_subscription_client_uses_luna_without_tools_or_sessions(tmp_path: Path):
    observed: dict[str, object] = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        prompt_arg = next(arg for arg in args if arg.startswith("@"))
        observed["prompt"] = Path(prompt_arg[1:]).read_text(encoding="utf-8")
        message = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "PONG"}],
                "usage": {
                    "input": 120,
                    "output": 4,
                    "cacheRead": 20,
                    "cacheWrite": 0,
                    "totalTokens": 144,
                    "cost": {"total": 0.0001},
                },
                "provider": "openai-codex",
                "model": "gpt-5.6-luna",
            },
        }
        return CompletedProcess(args, 0, stdout=json.dumps(message), stderr="")

    client = OmpSubscriptionClient(
        executable="omp",
        model="openai-codex/gpt-5.6-luna",
        thinking="low",
        temp_dir=tmp_path,
        run_command=fake_run,
    )

    result = client.generate("Return exactly PONG.", system="Follow the request.")

    assert result.text == "PONG"
    assert result.usage.input_tokens == 120
    assert result.usage.cache_read_tokens == 20
    assert result.usage.provider == "openai-codex"
    assert observed["prompt"] == "Return exactly PONG."
    args = observed["args"]
    assert "--no-tools" in args
    assert "--no-session" in args
    assert "--no-skills" in args
    assert "--no-rules" in args
    assert "openai-codex/gpt-5.6-luna" in args
