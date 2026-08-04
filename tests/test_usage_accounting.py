"""Tests for generation and retrieval accounting separation."""

from src.cost_tracker import CostTracker


def test_tracker_separates_generation_and_embedding_requests():
    tracker = CostTracker(max_dollars=10, warn_at=9)
    start = tracker.snapshot()

    tracker.record_generation(
        provider="openai-codex",
        model="gpt-5.6-luna",
        input_tokens=100,
        output_tokens=10,
        cache_read_tokens=20,
        cost_usd=0.001,
    )
    tracker.record_embedding(
        provider="openai",
        model="text-embedding-3-small",
        input_tokens=500,
        cost_usd=0.00001,
        exact=True,
    )

    delta = tracker.delta(start)
    assert delta.generation_calls == 1
    assert delta.embedding_calls == 1
    assert delta.generation_input_tokens == 100
    assert delta.embedding_input_tokens == 500
    assert delta.cost_usd == 0.00101
