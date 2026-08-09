"""
streaming.clock
===============

A single knob that controls how stream-time maps to wall-clock time, so the
same simulation can run live, accelerated for a demo, or with no sleeping at
all for deterministic tests.

    StreamClock(speed=1)     -> real time      (1 stream-second = 1 real second)
    StreamClock(speed=60)    -> 60x accelerated (1 stream-minute = 1 real second)
    StreamClock(speed=0)     -> no sleep       (as fast as possible; for tests)
"""
from __future__ import annotations

import time


class StreamClock:
    def __init__(self, speed: float = 1.0):
        # speed <= 0  => never sleep (deterministic / as-fast-as-possible)
        self.speed = speed
        self._t0_wall = time.monotonic()

    def sleep(self, stream_seconds: float) -> None:
        """Block for `stream_seconds` of stream time, scaled by speed."""
        if self.speed and self.speed > 0 and stream_seconds > 0:
            time.sleep(stream_seconds / self.speed)

    @property
    def real_elapsed(self) -> float:
        return time.monotonic() - self._t0_wall
