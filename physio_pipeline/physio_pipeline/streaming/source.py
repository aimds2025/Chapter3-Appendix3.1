"""
streaming.source
================

Turns a set of devices into a continuous, time-ordered stream of EdgePackets,
interleaved across patients the way a real ingestion endpoint would see them.

Features:
  * multi-patient interleave, ordered by stream timestamp
  * scheduled mid-stream events (e.g. onset of tachycardia at t=20s, or a
    poisoned window injected at t=30s)
  * optional jitter and late/out-of-order delivery, so the consumer can't
    assume perfectly ordered arrival

Each device emits one window every `window_s` of stream time. The generator
yields (event_ts, arrival_ts, EdgePacket): event_ts is when the window occurred
at the bedside; arrival_ts is when it reaches the bus (>= event_ts if delayed).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.contracts import EdgePacket
from ..layer1_edge.edge import EdgeDevice


@dataclass
class StreamEvent:
    at_s: float
    patient_id: str
    apply: Callable[[EdgeDevice], None]
    label: str = ""
    fired: bool = False


def onset(hr_low: float, hr_high: float) -> Callable[[EdgeDevice], None]:
    """Event action: shift a patient's heart-rate range (e.g. into tachycardia)."""
    def _apply(dev: EdgeDevice) -> None:
        dev.hr_range = (hr_low, hr_high)
    return _apply


def poison_next() -> Callable[[EdgeDevice], None]:
    """Event action: mark the device so its next window is a fabricated outlier."""
    def _apply(dev: EdgeDevice) -> None:
        setattr(dev, "_poison_next", True)
    return _apply


class StreamSource:
    def __init__(self, devices: list[EdgeDevice], window_s: float = 4.0,
                 duration_s: float = 60.0, jitter_s: float = 0.0,
                 late_prob: float = 0.0, late_max_s: float = 0.0,
                 seed: int = 0):
        self.devices = devices
        self.window_s = window_s
        self.duration_s = duration_s
        self.jitter_s = jitter_s
        self.late_prob = late_prob
        self.late_max_s = late_max_s
        self.rng = np.random.default_rng(seed)
        self._events: list[StreamEvent] = []

    def schedule(self, at_s: float, patient_id: str,
                 apply: Callable[[EdgeDevice], None], label: str = "") -> None:
        self._events.append(StreamEvent(at_s, patient_id, apply, label))

    def _emit_window(self, dev: EdgeDevice, event_ts: float) -> EdgePacket:
        frame = dev.acquire(duration_s=self.window_s)
        if getattr(dev, "_poison_next", False):
            # fabricate an erratic-rhythm outlier in the ECG channel
            n = frame.samples.shape[1]
            tt = np.arange(n) / dev.fs_hz
            seg = np.zeros(n)
            bt = 0.1
            while bt < n / dev.fs_hz:
                seg += np.exp(-0.5 * ((tt - bt) / 0.020) ** 2)
                bt += float(self.rng.choice([0.21, 0.22, 1.70]))
            frame.samples[0] = seg + self.rng.normal(0, 0.02, n)
            dev._poison_next = False
        return dev.emit(frame)

    def stream(self) -> Iterator[tuple[float, float, EdgePacket]]:
        """Yield (event_ts, arrival_ts, packet) in arrival order."""
        # min-heap keyed by arrival_ts; seed one window per device
        heap: list[tuple[float, float, int]] = []
        next_event_ts = {i: 0.0 for i in range(len(self.devices))}
        for i in range(len(self.devices)):
            heapq.heappush(heap, (0.0, 0.0, i))

        while heap:
            arrival_ts, event_ts, i = heapq.heappop(heap)
            if event_ts >= self.duration_s:
                continue
            dev = self.devices[i]

            # fire any scheduled events that are due for this patient
            for ev in self._events:
                if (not ev.fired and ev.patient_id == dev.patient_id
                        and event_ts >= ev.at_s):
                    ev.apply(dev)
                    ev.fired = True

            packet = self._emit_window(dev, event_ts)
            yield event_ts, arrival_ts, packet

            # schedule this device's next window
            nxt = event_ts + self.window_s
            jitter = (self.rng.uniform(-self.jitter_s, self.jitter_s)
                      if self.jitter_s else 0.0)
            arr = nxt + jitter
            if self.late_prob and self.rng.random() < self.late_prob:
                arr += self.rng.uniform(0, self.late_max_s)   # delayed delivery
            heapq.heappush(heap, (max(arr, nxt), nxt, i))

    @property
    def scheduled_events(self) -> list[StreamEvent]:
        return self._events
