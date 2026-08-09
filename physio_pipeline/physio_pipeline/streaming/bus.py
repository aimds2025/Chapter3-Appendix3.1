"""
streaming.bus
=============

A bounded in-process buffer between producer (StreamSource) and consumer. When
the consumer can't keep up, the buffer fills and the drop policy decides what
happens -- this is where backpressure becomes visible. It stands in for a
Kafka topic's finite retention / a bounded channel.

Policies:
  * "block"        -- caller waits (lossless; back-pressures the producer)
  * "drop_newest"  -- reject the incoming item (keep buffered history)
  * "drop_oldest"  -- evict the oldest buffered item to admit the new one
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusMetrics:
    admitted: int = 0
    dropped: int = 0
    max_depth: int = 0

    @property
    def drop_rate(self) -> float:
        total = self.admitted + self.dropped
        return self.dropped / total if total else 0.0


class BoundedBus:
    def __init__(self, maxsize: int = 32, policy: str = "drop_oldest"):
        if policy not in ("block", "drop_newest", "drop_oldest"):
            raise ValueError(f"unknown policy {policy!r}")
        self.maxsize = maxsize
        self.policy = policy
        self._q: deque = deque()
        self.metrics = BusMetrics()

    def put(self, item: Any) -> bool:
        """Offer an item. Returns True if admitted, False if dropped."""
        if len(self._q) >= self.maxsize:
            if self.policy == "drop_newest":
                self.metrics.dropped += 1
                return False
            if self.policy == "drop_oldest":
                self._q.popleft()
                self.metrics.dropped += 1
                # fall through and admit the new item
            # "block" is cooperative here: the single-threaded runner simply
            # processes one item before offering again, so we admit.
        self._q.append(item)
        self.metrics.admitted += 1
        self.metrics.max_depth = max(self.metrics.max_depth, len(self._q))
        return True

    def get(self) -> Any:
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)

    def empty(self) -> bool:
        return not self._q
