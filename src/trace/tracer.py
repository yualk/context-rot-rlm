"""Trace tree data structures for recording controller execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceNode:
    """A node in the execution trace tree."""

    action: str = ""
    input: str = ""
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[TraceNode] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def add_child(self, child: TraceNode) -> TraceNode:
        self.children.append(child)
        return child

    def finish(self) -> None:
        self.end_time = time.time()
        for child in self.children:
            if child.end_time is None:
                child.finish()


    @property
    def duration(self) -> float | None:
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def max_depth(self, current: int = 0) -> int:
        if not self.children:
            return current
        return max(c.max_depth(current + 1) for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)
    def walk(self):
        """Yield this node and all descendants depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action": self.action,
            "input": self.input or "",
            "output": self.output or "",
        }
        if self.metadata:
            d["metadata"] = self.metadata
        if self.duration is not None:
            d["duration_s"] = round(self.duration, 3)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d
