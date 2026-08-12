"""Contracts for the sparse and information-dense diagnostic benchmarks."""

from __future__ import annotations

from benchmarks.diagnostic import (
    OolongScorer,
    generate_dense_aggregation,
    generate_sniah,
)


def test_sniah_reuses_same_task_identity_across_context_lengths():
    samples = generate_sniah(
        context_lengths=[2_000, 4_000],
        tasks_per_length=2,
        seed=7,
        filler="Natural prose sentence one. Natural prose sentence two. ",
    )

    by_identity = {}
    for sample in samples:
        by_identity.setdefault(sample.task_identity, []).append(sample)

    assert len(by_identity) == 2
    assert all(len(group) == 2 for group in by_identity.values())
    assert all(len({s.answer for s in group}) == 1 for group in by_identity.values())
    assert all(len({s.needle_position for s in group}) == 1 for group in by_identity.values())


def test_dense_aggregation_requires_processing_many_records():
    sample = generate_dense_aggregation(num_records=200, seed=11)

    assert len(sample.document.splitlines()) == 200
    assert sample.answer.isdigit()
    assert sample.metadata["semantic_work"] == "linear"


def test_oolong_numeric_scorer_matches_paper_partial_credit():
    scorer = OolongScorer()

    assert scorer.score("10", "[10]") == 1.0
    assert scorer.score("11", "[10]") == 0.75
    assert scorer.score("12", "[10]") == 0.75**2
    assert scorer.score("unknown", "[10]") == 0.0
