"""Reproducible paired runner for sparse and dense long-context diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmarks.diagnostic import (
    DiagnosticSample,
    OOLONG_REVISION,
    OolongScorer,
    generate_dense_aggregation,
    generate_sniah,
    generate_recursive_aggregation,
    load_oolong,
)
from experiments.result_store import ResultStore, configuration_hash
from src.config import PROJECT_ROOT, settings
from src.controllers.base import BaseController
from src.controllers.fullcontext import ContextWindowExceeded, FullContextController
from src.controllers.rag_baseline import RAGController
from src.controllers.rlm_controller import RLMController
from src.cost_tracker import tracker
from src.environment.document_store import DocumentStore
from src.model_client import ModelClient, SubscriptionLimitError, get_default_client
from src.trace.trace_viewer import export_trace

RESULTS_DIR = PROJECT_ROOT / settings.output_dir
PRIMARY_METHODS = (
    "fullcontext",
    "rag",
    "rlm_symbolic",
    "rlm_depth0",
    "rlm_depth1",
)
ControllerFactory = Callable[[str], BaseController]


def get_controller(method: str, *, client: ModelClient | None = None) -> BaseController:
    """Construct comparable controllers around one generation client."""
    shared = client or get_default_client()
    if method == "fullcontext":
        return FullContextController(client=shared)
    if method == "rag":
        return RAGController(client=shared)
    if method == "rlm_symbolic":
        return RLMController(client=shared, max_depth=0, allow_subcalls=False)
    if method == "rlm_depth0":
        return RLMController(client=shared, max_depth=0)
    if method == "rlm_depth1":
        return RLMController(client=shared, max_depth=1)
    raise ValueError(f"Unknown method {method!r}; choose from {PRIMARY_METHODS}.")


class ExperimentRunner:
    """Run paired sample/method cells and persist each cell immediately."""

    def __init__(
        self,
        *,
        samples: Iterable[DiagnosticSample],
        methods: Iterable[str],
        output_path: str | Path,
        benchmark: str,
        seed: int = 42,
        client: ModelClient | None = None,
        controller_factory: ControllerFactory | None = None,
        trace_dir: str | Path | None = None,
        manifest_extra: dict[str, Any] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.methods = tuple(methods)
        unknown = set(self.methods).difference(PRIMARY_METHODS)
        if unknown and controller_factory is None:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("Methods must be unique.")
        self.benchmark = benchmark
        self.seed = seed
        self.client = client or (None if controller_factory else get_default_client())
        self.controller_factory = controller_factory or (
            lambda method: get_controller(method, client=self.client)
        )
        self.controllers = {method: self.controller_factory(method) for method in self.methods}
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None

        scientific_config = {
            "benchmark": benchmark,
            "methods": list(self.methods),
            "model_provider": settings.model_provider,
            "model": settings.model_generation,
            "thinking": settings.model_thinking,
            "context_window_tokens": settings.model_context_window,
            "fullcontext_max_tokens": settings.fullcontext_max_tokens,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "rag_top_k": settings.rag_top_k,
            "rlm": {
                "max_iterations": settings.rlm_max_iterations,
                "max_subcalls": settings.rlm_max_subcalls,
                "history_chars": settings.rlm_history_chars,
            },
            "seed": seed,
            "source_revision": _source_revision(),
            **(manifest_extra or {}),
        }
        config_hash = configuration_hash(scientific_config)
        manifest = {
            **scientific_config,
            "operational": {
                "subscription_call_limit": settings.subscription_call_limit,
                "sample_count": len(self.samples),
                "result_format": "jsonl",
                "method_order": "deterministically shuffled within sample",
            },
        }
        self.store = ResultStore(output_path, config_hash=config_hash, manifest=manifest)

    def run(self) -> list[dict[str, Any]]:
        for sample in self.samples:
            methods = list(self.methods)
            random.Random(_stable_seed(self.seed, sample.sample_id)).shuffle(methods)
            for method in methods:
                if not self.store.pending(method, [sample.sample_id]):
                    continue
                stop = self._run_cell(sample, method)
                if stop:
                    return self._configuration_rows()
        return self._configuration_rows()

    def _run_cell(self, sample: DiagnosticSample, method: str) -> bool:
        controller = self.controllers[method]
        store = DocumentStore()
        store.ingest(sample.document, doc_id=sample.sample_id)
        usage_start = tracker.snapshot()
        started = time.perf_counter()
        status = "ok"
        error = ""
        prediction = ""
        confidence = 0.0
        controller_metadata: dict[str, Any] = {}
        trace = None
        stop = False
        try:
            result = controller.answer(sample.question, store)
            prediction = result.answer
            confidence = result.confidence
            controller_metadata = result.metadata
            trace = result.trace
        except ContextWindowExceeded as exc:
            status = "context_window_exceeded"
            error = str(exc)
        except SubscriptionLimitError as exc:
            status = "subscription_limit"
            error = str(exc)
            stop = True
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        duration_s = time.perf_counter() - started
        usage = tracker.delta(usage_start)
        score = _score(sample, prediction) if status == "ok" else 0.0
        row = {
            "sample_id": sample.sample_id,
            "task_identity": sample.task_identity,
            "method": method,
            "status": status,
            "question": sample.question,
            "reference": sample.answer,
            "prediction": prediction,
            "score": score,
            "confidence": confidence,
            "duration_s": duration_s,
            "context_length_tokens": sample.context_length_tokens,
            "needle_position": sample.needle_position,
            "generation_input_tokens": usage.generation_input_tokens,
            "generation_output_tokens": usage.generation_output_tokens,
            "generation_cache_read_tokens": usage.generation_cache_read_tokens,
            "generation_cache_write_tokens": usage.generation_cache_write_tokens,
            "generation_calls": usage.generation_calls,
            "embedding_input_tokens": usage.embedding_input_tokens,
            "embedding_calls": usage.embedding_calls,
            "billed_cost_usd": usage.cost_usd,
            "estimated_cost_usd": usage.estimated_cost_usd,
            "usage_exact": usage.exact,
            "error": error,
            "sample_metadata": sample.metadata,
            "controller_metadata": controller_metadata,
        }
        self.store.append(row)
        if self.trace_dir is not None and trace is not None:
            export_trace(trace, self.trace_dir / f"{sample.sample_id}.{method}.json")
        return stop

    def _configuration_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self.store.rows if row.get("config_hash") == self.store.config_hash
        ]


def _score(sample: DiagnosticSample, prediction: str) -> float:
    benchmark = sample.metadata.get("benchmark")
    if benchmark in {"oolong", "dense_aggregation", "recursive_aggregation"}:
        return OolongScorer().score(
            prediction,
            sample.answer,
            sample.metadata.get("answer_type"),
        )
    return float(_normalize_answer(prediction) == _normalize_answer(sample.answer))


def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().casefold().split()).strip(".,;:[](){}\"'")


def _stable_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _source_revision() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = "unknown"

    digest = hashlib.sha256()
    roots = ("analysis", "benchmarks", "experiments", "src", "tests")
    files = [
        path
        for root in roots
        for path in (PROJECT_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    files.extend((PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "requirements.txt"))
    for path in sorted(files):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return f"{head}-tree-{digest.hexdigest()[:12]}"


def _build_samples(args: argparse.Namespace) -> tuple[list[DiagnosticSample], dict[str, Any]]:
    if args.benchmark == "sniah":
        lengths = args.context_lengths or settings.benchmark_cfg["diagnostic"]["sniah_context_lengths"]
        samples = generate_sniah(
            context_lengths=lengths,
            tasks_per_length=args.tasks,
            seed=args.seed,
        )
        return samples, {"context_lengths": lengths, "tasks_per_length": args.tasks}
    if args.benchmark == "dense":
        sizes = args.record_counts or settings.benchmark_cfg["diagnostic"]["dense_record_counts"]
        samples = [
            generate_dense_aggregation(num_records=size, seed=args.seed + task)
            for size in sizes
            for task in range(args.tasks)
        ]
        return samples, {"record_counts": sizes, "tasks_per_size": args.tasks}
    if args.benchmark == "recursive":
        samples = [
            generate_recursive_aggregation(
                num_sections=args.sections,
                records_per_section=args.records_per_section,
                seed=args.seed + task,
            )
            for task in range(args.tasks)
        ]
        return samples, {
            "num_sections": args.sections,
            "records_per_section": args.records_per_section,
            "tasks": args.tasks,
        }
    samples = load_oolong(
        split=args.split,
        max_samples=args.max_samples,
        revision=OOLONG_REVISION,
        context_lengths=args.context_lengths,
    )
    return samples, {
        "dataset": "oolongbench/oolong-synth",
        "dataset_revision": OOLONG_REVISION,
        "split": args.split,
        "max_samples": args.max_samples,
        "context_lengths": args.context_lengths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("sniah", "dense", "recursive", "oolong"))
    parser.add_argument("--methods", nargs="+", choices=PRIMARY_METHODS, default=list(PRIMARY_METHODS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--context-lengths", type=int, nargs="+")
    parser.add_argument("--record-counts", type=int, nargs="+")
    parser.add_argument("--sections", type=int, default=4)
    parser.add_argument("--records-per-section", type=int, default=100)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, benchmark_config = _build_samples(args)
    output = args.output or RESULTS_DIR / f"{args.benchmark}.jsonl"
    runner = ExperimentRunner(
        samples=samples,
        methods=args.methods,
        output_path=output,
        benchmark=args.benchmark,
        seed=args.seed,
        trace_dir=args.trace_dir,
        manifest_extra={"benchmark_config": benchmark_config},
    )
    rows = runner.run()
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(json.dumps({"result_path": str(output), "rows": len(rows), "statuses": statuses}, indent=2))
    return 0 if statuses.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
