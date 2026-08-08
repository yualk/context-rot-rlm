"""Configuration loader: merges config.yaml with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml() -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_YAML: dict[str, Any] = _load_yaml()


class Settings(BaseSettings):
    # Subscription-backed generation
    model_provider: str = _YAML["models"]["provider"]
    model_generation: str = _YAML["models"]["generation"]
    model_thinking: str = _YAML["models"]["thinking"]
    model_context_window: int = _YAML["models"]["context_window_tokens"]
    model_timeout_s: int = _YAML["models"]["timeout_s"]
    omp_executable: str = "omp"
    subscription_call_limit: int = _YAML["subscription"]["call_limit"]


    # Budget
    max_dollars: float = _YAML["budget"]["max_dollars"]
    warn_at_dollars: float = _YAML["budget"]["warn_at_dollars"]

    # Chunking
    chunk_size: int = _YAML["chunking"]["chunk_size"]
    chunk_overlap: int = _YAML["chunking"]["chunk_overlap"]
    min_chunk_size: int = _YAML["chunking"]["min_chunk_size"]

    # Retrieval
    bm25_top_k: int = _YAML["retrieval"]["bm25_top_k"]
    rag_top_k: int = _YAML["retrieval"]["rag_top_k"]

    # RLM
    rlm_max_depth: int = _YAML["rlm"]["max_depth"]
    rlm_max_iterations: int = _YAML["rlm"]["max_iterations"]
    rlm_max_subcalls: int = _YAML["rlm"]["max_subcalls"]
    rlm_history_chars: int = _YAML["rlm"]["history_chars"]
    rlm_stdout_chars: int = _YAML["rlm"]["stdout_chars"]


    # Baselines
    fullcontext_max_tokens: int = _YAML["fullcontext"]["max_input_tokens"]

    # Benchmarks
    benchmark_cfg: dict[str, Any] = _YAML["benchmarks"]

    # Experiment
    seed: int = _YAML["experiment"]["seed"]
    methods: list[str] = _YAML["experiment"]["methods"]
    output_dir: str = _YAML["experiment"]["output_dir"]

    model_config = {"env_prefix": "", "extra": "ignore"}


settings = Settings()
