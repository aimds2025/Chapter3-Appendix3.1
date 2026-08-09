"""Layer 3 - Stream ingestion: Kafka-style log, schema registry, replay."""
from .ingestion import SchemaRegistry, StreamIngestion

__all__ = ["StreamIngestion", "SchemaRegistry"]
