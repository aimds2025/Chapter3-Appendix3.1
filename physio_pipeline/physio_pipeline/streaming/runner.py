"""
streaming.runner
================

Drives a StreamSource through a BoundedBus into the live consumer path
(Layer 2 authorize -> Layer 3 ingest -> Layer 4 online alarms + online poison
guard), paced by a StreamClock, and finally persists a consolidated micro-batch
to Layer 5.

Returns a StreamResult with the things only a *stream* can show: throughput,
bus drops under load, and detection latency (how long after a simulated onset
the first alarm fires).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.contracts import Alarm, CoresetPoint, ProcessedBatch, StreamRecord
from ..crosscutting.audit import AuditLog
from ..crosscutting.iam import IAM
from ..crosscutting.privacy_budget import PrivacyBudgetLedger
from ..layer2_perimeter import ZeroTrustPerimeter
from ..layer3_ingestion import StreamIngestion
from ..layer4_processing import embed
from ..layer5_storage import StorageLayer
from .bus import BoundedBus
from .clock import StreamClock
from .online_guard import OnlineRobustGuard
from .source import StreamSource


@dataclass
class StreamResult:
    produced: int = 0
    consumed: int = 0
    dropped: int = 0
    max_depth: int = 0
    drop_rate: float = 0.0
    coreset_admitted: int = 0
    vetoed: int = 0
    alarms: list[tuple[float, str, str]] = field(default_factory=list)  # (ts,pid,kind)
    onsets: list[tuple[float, str, str]] = field(default_factory=list)
    detection_latency_s: dict[str, float] = field(default_factory=dict)
    real_elapsed_s: float = 0.0
    throughput_pps: float = 0.0
    audit_events: int = 0
    audit_chain_head: str = ""

    def summary(self) -> str:
        return (f"produced={self.produced} consumed={self.consumed} "
                f"dropped={self.dropped} ({self.drop_rate:.1%}) "
                f"max_depth={self.max_depth} | coreset={self.coreset_admitted} "
                f"vetoed={self.vetoed} | alarms={len(self.alarms)}")


class StreamRunner:
    def __init__(self, source: StreamSource, clock: StreamClock | None = None,
                 bus: BoundedBus | None = None, sqi_floor: float = 0.2,
                 z_threshold: float = 8.0):
        self.source = source
        self.clock = clock if clock is not None else StreamClock(speed=0.0)
        # NOTE: BoundedBus defines __len__, so an empty bus is falsy -- `bus or ...`
        # would silently discard a caller-supplied empty bus. Use an explicit
        # None check.
        self.bus = bus if bus is not None else BoundedBus(maxsize=32,
                                                          policy="drop_oldest")
        self.sqi_floor = sqi_floor

        self.audit = AuditLog()
        self.iam = IAM()
        self.ledger = PrivacyBudgetLedger()
        self.perimeter = ZeroTrustPerimeter(self.iam, self.audit)
        self.ingestion = StreamIngestion(self.audit)
        self.storage = StorageLayer(self.ledger)
        self.guard = OnlineRobustGuard(n_features=4, z_threshold=z_threshold)

        for dev in source.devices:
            self.iam.register(f"spiffe://edge/{dev.device_id}", {"device"})

    def run(self, load_spike: tuple[float, float, float] | None = None
            ) -> StreamResult:
        """
        load_spike = (start_s, end_s, drain_per_put): during [start,end] the
        consumer drains fewer items per produced item, so the bus fills and the
        drop policy kicks in -- a visible backpressure episode.
        """
        res = StreamResult()
        res.onsets = [(e.at_s, e.patient_id, e.label)
                      for e in self.source.scheduled_events]
        consumed_records: list[StreamRecord] = []
        credits = 0.0
        prev_arrival = 0.0

        def consume_one() -> None:
            if self.bus.empty():
                return
            packet = self.bus.get()
            try:
                packet = self.perimeter.authorize(packet)      # L2
            except Exception:
                return
            rec = self.ingestion.produce(packet)               # L3
            consumed_records.append(rec)
            res.consumed += 1

            # ---- L4 online: alarms (SQI-gated) ----
            e = embed(packet)
            hr = float(e[0] * 200.0)
            ev_ts = getattr(packet, "_event_ts", 0.0)
            if packet.sqi < self.sqi_floor:
                res.alarms.append((ev_ts, packet.patient_id, "signal_loss"))
            elif hr > 120:
                res.alarms.append((ev_ts, packet.patient_id, "tachycardia"))
                self._record_latency(res, packet.patient_id, ev_ts)
            elif 0 < hr < 45:
                res.alarms.append((ev_ts, packet.patient_id, "bradycardia"))
                self._record_latency(res, packet.patient_id, ev_ts)

            # ---- L4 online: poison guard for coreset admission ----
            if packet.sqi >= self.sqi_floor:
                accept, z = self.guard.check(e)
                if accept:
                    res.coreset_admitted += 1
                else:
                    res.vetoed += 1
                    self.audit.record("layer4", "online_veto", packet.patient_id,
                                      ok=False, max_z=round(z, 2))

        for event_ts, arrival_ts, packet in self.source.stream():
            setattr(packet, "_event_ts", event_ts)
            # pace to arrival time (scaled by clock speed)
            gap = max(0.0, arrival_ts - prev_arrival)
            self.clock.sleep(gap)
            prev_arrival = arrival_ts

            admitted = self.bus.put(packet)
            res.produced += 1

            drain = 1.0
            if load_spike and load_spike[0] <= event_ts < load_spike[1]:
                drain = load_spike[2]            # consumer falls behind
            credits += drain
            # Clamp so surplus credit from quiet periods can't carry over and
            # silently cancel a later backpressure episode.
            credits = min(credits, 1.0)
            while credits >= 1.0 and not self.bus.empty():
                consume_one()
                credits -= 1.0

        # drain whatever remains after the stream ends
        while not self.bus.empty():
            consume_one()

        # ---- L5 persist a consolidated micro-batch ----
        if consumed_records:
            coreset = [CoresetPoint(patient_id=r.packet.patient_id,
                                    embedding=embed(r.packet), weight=1.0,
                                    source_offset=r.offset)
                       for r in consumed_records[: self.guard.warmup]]
            batch = ProcessedBatch(alarms=[], coreset=coreset, dq_metrics={},
                                   rejected=res.vetoed)
            self.storage.persist("stream", consumed_records, batch)

        m = self.bus.metrics
        res.dropped = m.dropped
        res.max_depth = m.max_depth
        res.drop_rate = m.drop_rate
        res.real_elapsed_s = self.clock.real_elapsed
        res.throughput_pps = (res.consumed / res.real_elapsed_s
                              if res.real_elapsed_s > 0 else float("inf"))
        res.audit_events = len(self.audit)
        res.audit_chain_head = self.audit.head
        return res

    @staticmethod
    def _record_latency(res: StreamResult, pid: str, ev_ts: float) -> None:
        if pid in res.detection_latency_s:
            return
        for at_s, opid, label in res.onsets:
            # only a genuine rate ONSET has a meaningful "detection latency";
            # a poisoned-window event is not a physiological onset.
            if opid == pid and "onset" in label.lower() and ev_ts >= at_s:
                res.detection_latency_s[pid] = round(ev_ts - at_s, 2)
                return
