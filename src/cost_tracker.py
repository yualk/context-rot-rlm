"""Exact, category-separated model usage accounting."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class ModelUsage:
    provider: str
    model: str
    kind: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    billed_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    exact: bool = True


@dataclass(frozen=True)
class UsageSnapshot:
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    generation_cache_read_tokens: int = 0
    generation_cache_write_tokens: int = 0
    embedding_input_tokens: int = 0
    generation_calls: int = 0
    embedding_calls: int = 0
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    exact: bool = True


@dataclass
class CostTracker:
    max_dollars: float = field(default_factory=lambda: settings.max_dollars)
    warn_at: float = field(default_factory=lambda: settings.warn_at_dollars)
    usage: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(item.billed_cost_usd for item in self.usage.values())

    @property
    def estimated_cost(self) -> float:
        with self._lock:
            return sum(item.estimated_cost_usd for item in self.usage.values())

    def record_generation(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        estimated_cost_usd: float = 0.0,
        exact: bool = True,
    ) -> None:
        self._record(
            kind="generation",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            estimated_cost_usd=estimated_cost_usd,
            exact=exact,
        )

    def record_embedding(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        cost_usd: float = 0.0,
        estimated_cost_usd: float = 0.0,
        exact: bool = True,
    ) -> None:
        self._record(
            kind="embedding",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=cost_usd,
            estimated_cost_usd=estimated_cost_usd,
            exact=exact,
        )

    def _record(
        self,
        *,
        kind: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        cost_usd: float,
        estimated_cost_usd: float,
        exact: bool,
    ) -> None:
        key = f"{kind}:{provider}:{model}"
        with self._lock:
            item = self.usage.setdefault(
                key,
                ModelUsage(provider=provider, model=model, kind=kind),
            )
            item.input_tokens += int(input_tokens)
            item.output_tokens += int(output_tokens)
            item.cache_read_tokens += int(cache_read_tokens)
            item.cache_write_tokens += int(cache_write_tokens)
            item.calls += 1
            item.billed_cost_usd += float(cost_usd)
            item.estimated_cost_usd += float(estimated_cost_usd)
            item.exact = item.exact and exact
            billed_total = sum(value.billed_cost_usd for value in self.usage.values())
        if billed_total >= self.warn_at:
            logger.warning("Budget warning: $%.2f / $%.2f", billed_total, self.max_dollars)

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            generation = [item for item in self.usage.values() if item.kind == "generation"]
            embedding = [item for item in self.usage.values() if item.kind == "embedding"]
            return UsageSnapshot(
                generation_input_tokens=sum(item.input_tokens for item in generation),
                generation_output_tokens=sum(item.output_tokens for item in generation),
                generation_cache_read_tokens=sum(item.cache_read_tokens for item in generation),
                generation_cache_write_tokens=sum(item.cache_write_tokens for item in generation),
                embedding_input_tokens=sum(item.input_tokens for item in embedding),
                generation_calls=sum(item.calls for item in generation),
                embedding_calls=sum(item.calls for item in embedding),
                cost_usd=sum(item.billed_cost_usd for item in self.usage.values()),
                estimated_cost_usd=sum(item.estimated_cost_usd for item in self.usage.values()),
                exact=all(item.exact for item in self.usage.values()),
            )

    def delta(self, start: UsageSnapshot) -> UsageSnapshot:
        current = self.snapshot()
        return UsageSnapshot(
            generation_input_tokens=current.generation_input_tokens - start.generation_input_tokens,
            generation_output_tokens=current.generation_output_tokens - start.generation_output_tokens,
            generation_cache_read_tokens=(
                current.generation_cache_read_tokens - start.generation_cache_read_tokens
            ),
            generation_cache_write_tokens=(
                current.generation_cache_write_tokens - start.generation_cache_write_tokens
            ),
            embedding_input_tokens=current.embedding_input_tokens - start.embedding_input_tokens,
            generation_calls=current.generation_calls - start.generation_calls,
            embedding_calls=current.embedding_calls - start.embedding_calls,
            cost_usd=current.cost_usd - start.cost_usd,
            estimated_cost_usd=current.estimated_cost_usd - start.estimated_cost_usd,
            exact=current.exact and start.exact,
        )

    def check_budget(self) -> None:
        if self.total_cost >= self.max_dollars:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.2f} >= ${self.max_dollars:.2f}"
            )

    def summary(self) -> str:
        with self._lock:
            lines = ["=== Usage Summary ==="]
            for key, item in sorted(self.usage.items()):
                lines.append(
                    f"{key}: calls={item.calls}, input={item.input_tokens:,}, "
                    f"output={item.output_tokens:,}, billed=${item.billed_cost_usd:.4f}, "
                    f"estimated=${item.estimated_cost_usd:.4f}, exact={item.exact}"
                )
            lines.append(
                f"TOTAL: billed=${self.total_cost:.4f}, estimated=${self.estimated_cost:.4f}"
            )
            return "\n".join(lines)


tracker = CostTracker()
