"""
crosscutting.tracing
====================

Minimal OpenTelemetry-style span context so layer interactions are observable.
Production swap-in: the real `opentelemetry` SDK exporting to an OTLP collector.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attrs: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000.0


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs):
        sp = Span(name=name, start=time.time(), attrs=dict(attrs))
        try:
            yield sp
        finally:
            sp.end = time.time()
            self.spans.append(sp)

    def timeline(self) -> list[tuple[str, float]]:
        return [(s.name, round(s.duration_ms, 2)) for s in self.spans]
