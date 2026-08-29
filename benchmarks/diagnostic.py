"""Sparse-retrieval and dense-aggregation diagnostics for RLM evaluation."""

from __future__ import annotations

import ast
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

OOLONG_DATASET = "oolongbench/oolong-synth"
OOLONG_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"



@dataclass(frozen=True)
class DiagnosticSample:
    sample_id: str
    task_identity: str
    document: str
    question: str
    answer: str
    context_length_tokens: int
    needle_position: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


_TOPICS = (
    "amber",
    "cobalt",
    "comet",
    "crystal",
    "forest",
    "galaxy",
    "jade",
    "lemon",
    "mountain",
    "ocean",
    "onyx",
    "planet",
    "river",
    "silver",
)
_DEFAULT_FILLER = (
    "Researchers compared several historical records before publishing the archive. "
    "The surrounding discussion covers institutions, geography, and ordinary events. "
    "Each paragraph is independent and contains no hidden numerical instruction. "
)


def generate_sniah(
    *,
    context_lengths: list[int],
    tasks_per_length: int = 10,
    seed: int = 42,
    filler: str = _DEFAULT_FILLER,
) -> list[DiagnosticSample]:
    """Generate fixed S-NIAH tasks reused at every requested token length.

    Length is estimated at four characters per token. Task identity, answer, and
    insertion depth are sampled once and held constant across lengths, enabling
    paired scaling comparisons.
    """
    if not filler.strip():
        raise ValueError("filler must not be empty")
    rng = random.Random(seed)
    task_specs = [
        (
            f"task-{index:03d}",
            rng.choice(_TOPICS),
            rng.randint(100_000, 999_999),
            rng.uniform(0.1, 0.9),
        )
        for index in range(tasks_per_length)
    ]

    samples: list[DiagnosticSample] = []
    for identity, topic, value, position in task_specs:
        needle = f"The special magic {topic} number is: {value}."
        question = f"What is the special magic {topic} number? Return only the number."
        for length in context_lengths:
            target_chars = max(length * 4, len(needle) + 2)
            repeats = target_chars // len(filler) + 2
            haystack = (filler * repeats)[: target_chars - len(needle) - 1]
            insert_at = int(len(haystack) * position)
            document = f"{haystack[:insert_at]}\n{needle}\n{haystack[insert_at:]}"
            digest = hashlib.sha256(f"{identity}:{length}:{seed}".encode()).hexdigest()[:10]
            samples.append(
                DiagnosticSample(
                    sample_id=f"sniah-{length}-{digest}",
                    task_identity=identity,
                    document=document,
                    question=question,
                    answer=str(value),
                    context_length_tokens=length,
                    needle_position=position,
                    metadata={
                        "benchmark": "sniah",
                        "semantic_work": "constant",
                        "topic": topic,
                        "target_tokens": length,
                    },
                )
            )
    return samples


_CATEGORY_TEXTS = {
    "science": (
        "Researchers measured a new material under controlled laboratory conditions.",
        "Astronomers reported observations from a distant stellar system.",
        "A clinical study compared outcomes across two treatment groups.",
        "Biologists catalogued a newly observed behavior in migratory animals.",
    ),
    "business": (
        "The company revised its quarterly revenue forecast after the acquisition.",
        "Investors evaluated the merger and changes to operating margins.",
        "The retailer expanded distribution while reducing inventory costs.",
        "A manufacturer announced a new supply agreement with regional partners.",
    ),
    "sports": (
        "The team secured the championship after scoring in extra time.",
        "The coach changed the starting lineup before the final match.",
        "A runner broke the course record during the international meet.",
        "The league published its playoff schedule after the regular season.",
    ),
    "culture": (
        "The museum opened an exhibition of paintings from the early modern period.",
        "A novelist received an award for a historical work of fiction.",
        "The orchestra performed a newly restored composition at the festival.",
        "The theatre announced a revival featuring a new cast and director.",
    ),
}

_CATEGORY_RULES = {
    "science": "scientific research or empirical measurement",
    "business": "commercial, financial, or manufacturing activity",
    "sports": "athletic competition, coaching, or league activity",
    "culture": "artistic, literary, musical, museum, or theatre activity",
}


def generate_recursive_aggregation(
    *,
    num_sections: int = 4,
    records_per_section: int = 100,
    seed: int = 42,
) -> DiagnosticSample:
    """Generate section-local semantic counts that can be recursively reduced."""
    if num_sections < 2:
        raise ValueError("num_sections must be at least two")
    if records_per_section < 1:
        raise ValueError("records_per_section must be positive")

    rng = random.Random(seed)
    categories = list(_CATEGORY_TEXTS)
    target_order: list[str] = []
    while len(target_order) < num_sections:
        shuffled = categories[:]
        rng.shuffle(shuffled)
        target_order.extend(shuffled)

    places = ("Northport", "Lakeview", "Riverton", "Stonebridge", "Westhaven")
    sections: list[str] = []
    section_counts: list[int] = []
    section_targets = target_order[:num_sections]
    for section_index, target in enumerate(section_targets):
        lines = [
            f"=== SECTION {section_index:02d} ===",
            f"QUALIFYING RULE: Count records about {_CATEGORY_RULES[target]}.",
        ]
        count = 0
        for record_index in range(records_per_section):
            category = rng.choice(categories)
            sentence = rng.choice(_CATEGORY_TEXTS[category])
            place = rng.choice(places)
            year = rng.randint(1990, 2025)
            lines.append(
                f"record-{section_index:02d}-{record_index:05d}: "
                f"In {place} during {year}, {sentence} "
                f"Archive reference {rng.randint(1000, 9999)}."
            )
            count += int(category == target)
        sections.extend(lines)
        section_counts.append(count)

    document = "\n".join(sections)
    return DiagnosticSample(
        sample_id=f"recursive-{num_sections}-{records_per_section}-{seed}",
        task_identity=f"recursive-{seed}",
        document=document,
        question=(
            "Each section states its own qualifying rule. Count the qualifying "
            "records in every section and return their total sum. Analyze sections "
            "independently; when recursive analysis is available, use rlm_query on "
            "each section. Return only the integer."
        ),
        answer=str(sum(section_counts)),
        context_length_tokens=max(1, len(document.split())),
        metadata={
            "benchmark": "recursive_aggregation",
            "semantic_work": "hierarchical",
            "num_sections": num_sections,
            "records_per_section": records_per_section,
            "section_counts": section_counts,
            "section_targets": section_targets,
            "answer_type": "ANSWER_TYPE.NUMERIC",
        },
    )


def generate_dense_aggregation(*, num_records: int = 400, seed: int = 42) -> DiagnosticSample:
    """Generate an OOLONG-style linear semantic aggregation task."""
    if num_records < 1:
        raise ValueError("num_records must be positive")
    rng = random.Random(seed)
    categories = tuple(_CATEGORY_TEXTS)
    target = rng.choice(categories)
    records: list[str] = []
    count = 0
    for index in range(num_records):
        category = rng.choice(categories)
        text = rng.choice(_CATEGORY_TEXTS[category])
        records.append(f"record-{index:05d}: {text}")
        count += int(category == target)
    return DiagnosticSample(
        sample_id=f"dense-{num_records}-{seed}",
        task_identity=f"dense-{seed}",
        document="\n".join(records),
        question=f"How many records are about {target}? Return only the integer count.",
        answer=str(count),
        context_length_tokens=max(1, len(" ".join(records).split())),
        metadata={
            "benchmark": "dense_aggregation",
            "semantic_work": "linear",
            "num_records": num_records,
            "target_category": target,
        },
    )


def load_oolong(
    *,
    split: str = "validation",
    max_samples: int | None = 50,
    revision: str = OOLONG_REVISION,
    context_lengths: Iterable[int] | None = None,
) -> list[DiagnosticSample]:
    """Stream a pinned OOLONG split without downloading the multi-GB corpus."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to load OOLONG.") from exc

    allowed_lengths = set(context_lengths or ())
    stream = load_dataset(
        OOLONG_DATASET,
        split=split,
        revision=revision,
        streaming=True,
    )
    samples: list[DiagnosticSample] = []
    for row in stream:
        context_length = int(row["context_len"])
        if allowed_lengths and context_length not in allowed_lengths:
            continue
        sample_id = f"oolong-{split}-{row['id']}"
        samples.append(
            DiagnosticSample(
                sample_id=sample_id,
                task_identity=str(row["id"]),
                document=str(row["context_window_text"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                context_length_tokens=context_length,
                metadata={
                    "benchmark": "oolong",
                    "dataset": str(row["dataset"]),
                    "dataset_revision": revision,
                    "split": split,
                    "task_group": str(row["task_group"]),
                    "task": str(row["task"]),
                    "answer_type": str(row["answer_type"]),
                    "input_subset": str(row["input_subset"]),
                    "num_labels": int(row["num_labels"]),
                    "context_window_id": int(row["context_window_id"]),
                },
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class OolongScorer:
    """Faithful OOLONG synthetic scoring with numeric partial credit."""

    def score(
        self,
        prediction: str,
        gold: str,
        answer_type: str | None = None,
    ) -> float:
        gold_value = self._unwrap(gold)
        candidate = self._candidate(prediction)
        if candidate == gold_value:
            return 1.0
        comparison_labels = ("more common", "less common", "same frequency")
        if candidate in comparison_labels and candidate in gold_value:
            return 1.0

        numeric = answer_type == "ANSWER_TYPE.NUMERIC"
        gold_number = self._number(gold_value)
        prediction_number = self._number(candidate)
        if numeric or (answer_type is None and gold_number is not None):
            if gold_number is None or prediction_number is None:
                return 0.0
            return 0.75 ** abs(gold_number - prediction_number)
        return float(candidate.casefold() == gold_value.casefold())

    def _candidate(self, value: str) -> str:
        candidate = value.strip()
        if ":" in candidate:
            candidate = candidate.rsplit(":", 1)[-1].strip()
        candidate = candidate.replace("*", "").strip()
        if candidate.startswith("[") and candidate.endswith("]"):
            candidate = candidate[1:-1].strip()
        lowered = candidate.casefold()
        for label in ("more common", "less common", "same frequency"):
            if label in lowered:
                return label
        return candidate

    def _unwrap(self, value: str) -> str:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
        if isinstance(parsed, list) and len(parsed) == 1:
            return str(parsed[0]).strip()
        return value.strip()

    def _number(self, value: str) -> float | None:
        match = _NUMBER_RE.search(value)
        return float(match.group()) if match else None
