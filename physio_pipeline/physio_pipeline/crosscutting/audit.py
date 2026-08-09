"""
crosscutting.audit
==================

Tamper-evident audit trail feeding the SIEM. Every layer writes events here.
Events are hash-chained (each record commits to the previous head), which is
the mechanism behind the 21 CFR Part 11 immutability story in Layer 8.

Production swap-in: append events to Amazon OpenSearch / Elastic + an
append-only, hash-chained table in Postgres; ship to a SIEM (Sentinel/Splunk).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class AuditEvent:
    layer: str
    action: str
    actor: str
    ok: bool
    meta: dict
    prev_hash: str
    ts: float = field(default_factory=time.time)

    def digest(self) -> str:
        body = json.dumps(
            {
                "layer": self.layer, "action": self.action, "actor": self.actor,
                "ok": self.ok, "meta": self.meta, "prev": self.prev_hash,
                "ts": round(self.ts, 6),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(body).hexdigest()


class AuditLog:
    """Hash-chained, append-only audit log."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._head = self.GENESIS

    def record(self, layer: str, action: str, actor: str,
               ok: bool = True, **meta) -> str:
        evt = AuditEvent(layer, action, actor, ok, meta, prev_hash=self._head)
        self._head = evt.digest()
        self._events.append(evt)
        return self._head

    @property
    def head(self) -> str:
        return self._head

    def __len__(self) -> int:
        return len(self._events)

    def verify(self) -> bool:
        """Recompute the chain to detect any tampering."""
        prev = self.GENESIS
        for evt in self._events:
            if evt.prev_hash != prev:
                return False
            prev = evt.digest()
        return prev == self._head
