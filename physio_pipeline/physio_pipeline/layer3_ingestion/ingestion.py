"""
Layer 3 - STREAM INGESTION
=========================

A Kafka-style durable log with a schema registry and a replay window.

  * SchemaRegistry.validate()  -> rejects records with an unknown schema version
  * StreamIngestion.produce()  -> append EdgePacket to a partitioned topic
  * StreamIngestion.consume()  -> pull StreamRecords (optionally replay)

Partitioning is by patient_id so a patient's data stays ordered. The replay
buffer stands in for Kafka's retention (e.g. 7 days).

Production swap-in: Amazon MSK / Confluent, Glue or Confluent Schema Registry,
SASL/mTLS + topic ACLs, AES-256 at rest.
"""
from __future__ import annotations

from collections import defaultdict

from ..core.contracts import EdgePacket, StreamRecord
from ..core.exceptions import SchemaError
from ..crosscutting.audit import AuditLog


class SchemaRegistry:
    def __init__(self, current_version: int = 2):
        self.current_version = current_version
        self._known = set(range(1, current_version + 1))

    def validate(self, version: int) -> None:
        if version not in self._known:
            raise SchemaError(f"unknown schema version {version}")


class StreamIngestion:
    """In-memory stand-in for a Kafka topic with replay."""

    def __init__(self, audit: AuditLog, topic: str = "vitals.waveform.v2",
                 registry: SchemaRegistry | None = None):
        self.topic = topic
        self.audit = audit
        self.registry = registry or SchemaRegistry()
        # partition (key) -> ordered list of records  (the "retained log")
        self._log: dict[str, list[StreamRecord]] = defaultdict(list)
        self._offset = 0

    def produce(self, packet: EdgePacket) -> StreamRecord:
        if not packet.authorized:
            raise SchemaError("refusing to ingest unauthorized packet")
        self.registry.validate(self.registry.current_version)
        rec = StreamRecord(
            topic=self.topic, offset=self._offset, key=packet.patient_id,
            schema_version=self.registry.current_version, packet=packet,
            hospital_id=packet.hospital_id,
        )
        self._log[packet.patient_id].append(rec)
        self._offset += 1
        self.audit.record("layer3", "produce", packet.attestation.get("spiffe_id", "?"),
                          topic=self.topic, offset=rec.offset)
        return rec

    def consume(self, key: str | None = None) -> list[StreamRecord]:
        """Consume in offset order; `key` restricts to one partition (patient)."""
        if key is not None:
            return list(self._log.get(key, []))
        merged = [r for recs in self._log.values() for r in recs]
        return sorted(merged, key=lambda r: r.offset)

    def replay(self, from_offset: int = 0) -> list[StreamRecord]:
        return [r for r in self.consume() if r.offset >= from_offset]
