"""Analyze new JSONL experiments with paired, task-clustered comparisons."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from analysis.paired import paired_method_difference


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load complete JSONL rows, tolerating only a truncated final write."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"Invalid row at line {index + 1}")
        rows.append(value)
    return rows


def analyze_rows(
    rows: Iterable[dict[str, Any]],
    *,
    baseline: str = "rag",
    config_hash: str | None = None,
    condition_field: str | None = None,
    bootstrap_iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return coverage, descriptive means, and matched paired differences."""
    materialized = list(rows)
    hashes = {str(row.get("config_hash", "")) for row in materialized}
    if config_hash is None:
        if len(hashes) > 1:
            raise ValueError(
                "Result file contains multiple configurations; pass config_hash explicitly."
            )
        config_hash = next(iter(hashes), "")
    selected = [row for row in materialized if str(row.get("config_hash", "")) == config_hash]
    if not selected:
        raise ValueError(f"No rows found for configuration {config_hash!r}.")

    status_counts = Counter(str(row.get("status", "unknown")) for row in selected)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_method[str(row["method"])].append(row)

    method_summary: dict[str, dict[str, Any]] = {}
    for method, method_rows in sorted(by_method.items()):
        successful = [row for row in method_rows if row.get("status") == "ok"]
        method_summary[method] = {
            "attempted": len(method_rows),
            "successful": len(successful),
            "coverage": len(successful) / len(method_rows),
            "mean_score": fmean(float(row["score"]) for row in successful) if successful else None,
            "mean_duration_s": (
                fmean(float(row["duration_s"]) for row in successful) if successful else None
            ),
            "mean_generation_calls": (
                fmean(float(row["generation_calls"]) for row in successful)
                if successful
                else None
            ),
            "mean_generation_input_tokens": (
                fmean(float(row["generation_input_tokens"]) for row in successful)
                if successful
                else None
            ),
            "mean_embedding_calls": (
                fmean(float(row["embedding_calls"]) for row in successful)
                if successful
                else None
            ),
        }

    comparisons = []
    for treatment in sorted(set(by_method).difference({baseline})):
        try:
            paired = paired_method_difference(
                selected,
                baseline=baseline,
                treatment=treatment,
                iterations=bootstrap_iterations,
                seed=seed,
            )
        except ValueError:
            continue
        comparisons.append(asdict(paired))

    by_context: dict[str, list[dict[str, Any]]] = {}
    lengths = sorted({int(row["context_length_tokens"]) for row in selected})
    for length in lengths:
        subset = [row for row in selected if int(row["context_length_tokens"]) == length]
        context_comparisons = []
        for treatment in sorted(set(by_method).difference({baseline})):
            try:
                paired = paired_method_difference(
                    subset,
                    baseline=baseline,
                    treatment=treatment,
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
            except ValueError:
                continue
            context_comparisons.append(asdict(paired))
        by_context[str(length)] = context_comparisons

    by_condition: dict[str, list[dict[str, Any]]] = {}
    if condition_field is not None:
        condition_values = sorted({
            row.get("sample_metadata", {}).get(condition_field)
            for row in selected
            if row.get("sample_metadata", {}).get(condition_field) is not None
        })
        for value in condition_values:
            subset = [
                row
                for row in selected
                if row.get("sample_metadata", {}).get(condition_field) == value
            ]
            condition_comparisons = []
            for treatment in sorted(set(by_method).difference({baseline})):
                try:
                    paired = paired_method_difference(
                        subset,
                        baseline=baseline,
                        treatment=treatment,
                        iterations=bootstrap_iterations,
                        seed=seed,
                    )
                except ValueError:
                    continue
                condition_comparisons.append(asdict(paired))
            by_condition[str(value)] = condition_comparisons

    return {
        "config_hash": config_hash,
        "baseline": baseline,
        "rows": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "methods": method_summary,
        "paired_comparisons": comparisons,
        "paired_by_context_length": by_context,
        "condition_field": condition_field,
        "paired_by_condition": by_condition,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", default="rag")
    parser.add_argument("--config-hash")
    parser.add_argument("--condition-field")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    summary = analyze_rows(
        load_jsonl(args.results),
        baseline=args.baseline,
        config_hash=args.config_hash,
        condition_field=args.condition_field,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
