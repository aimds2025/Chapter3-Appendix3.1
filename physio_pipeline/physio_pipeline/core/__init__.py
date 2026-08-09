"""Core: shared data contracts and exceptions used by every layer."""
from .contracts import (
    Alarm,
    ComplianceReport,
    CoresetPoint,
    EdgePacket,
    ModelArtifact,
    ProcessedBatch,
    StorageReceipt,
    StreamRecord,
    WaveformFrame,
)
from .exceptions import (
    AuthorizationError,
    ComplianceViolation,
    PipelineError,
    PoisoningRejected,
    PrivacyBudgetExhausted,
    SchemaError,
)

__all__ = [
    "WaveformFrame", "EdgePacket", "StreamRecord", "CoresetPoint", "Alarm",
    "ProcessedBatch", "StorageReceipt", "ModelArtifact", "ComplianceReport",
    "PipelineError", "AuthorizationError", "SchemaError", "PoisoningRejected",
    "PrivacyBudgetExhausted", "ComplianceViolation",
]
