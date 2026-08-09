"""
core.contracts
==============

Typed data contracts that flow *between* layers. These dataclasses are the
"wire format" of the pipeline: each layer accepts the output type of the layer
before it and returns its own output type. Reading this file top-to-bottom is
the fastest way to understand how the layers interact.

Flow of the primary objects:

    WaveformFrame        (Layer 1 acquires)
        -> EdgePacket    (Layer 1 emits, Layer 2 authorizes)
        -> StreamRecord  (Layer 3 ingests/replays)
        -> ProcessedBatch(Layer 4 produces: alarms + coreset + DQ)
        -> StorageReceipt(Layer 5 persists)
        -> ModelArtifact (Layer 7 trains from the stored coreset)
        -> ComplianceReport (Layer 8 attests over the whole run)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Layer 1 <-> Layer 2/3
# --------------------------------------------------------------------------- #
@dataclass
class WaveformFrame:
    """Raw multi-channel acquisition window from a bedside device."""
    patient_id: str
    device_id: str
    fs_hz: int                       # sampling rate, e.g. 250
    channels: list[str]              # e.g. ["ECG_II", "PPG", ...]
    samples: np.ndarray              # shape (n_channels, n_samples)
    t_start: float = field(default_factory=time.time)
    hospital_id: str = "H000"        # tenant id (Section 9 multi-tenancy)

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]


@dataclass
class EdgePacket:
    """
    What Layer 1 sends over the wire after on-device processing.
    Payload is DWT-compressed; the raw frame never leaves the device.
    """
    patient_id: str
    device_id: str
    fs_hz: int
    channels: list[str]
    compressed: bytes                # DWT-compressed waveform payload
    prd_pct: float                   # percent-root-mean-square difference of compression
    qrs_indices: list[int]           # Pan-Tompkins R-peak sample indices
    sqi: float                       # signal quality index in [0, 1]
    t_start: float
    n_samples: int
    attestation: dict[str, Any]      # secure-boot / TPM-style signed envelope
    hospital_id: str = "H000"        # tenant id, propagated from the edge
    # populated by Layer 2 after authorization:
    zone: str | None = None
    authorized: bool = False


# --------------------------------------------------------------------------- #
# Layer 3 <-> Layer 4
# --------------------------------------------------------------------------- #
@dataclass
class StreamRecord:
    """A schema-validated record living on a Kafka-style topic."""
    topic: str
    offset: int
    key: str                         # partition key (patient_id)
    schema_version: int
    packet: EdgePacket
    ingest_ts: float = field(default_factory=time.time)
    hospital_id: str = "H000"        # tenant id, carried on the record


# --------------------------------------------------------------------------- #
# Layer 4 <-> Layer 5/7
# --------------------------------------------------------------------------- #
@dataclass
class CoresetPoint:
    """A single element selected by the streaming coreset (Layer 4)."""
    patient_id: str
    embedding: np.ndarray            # feature vector representing the window
    weight: float                    # coreset weight
    source_offset: int


@dataclass
class Alarm:
    patient_id: str
    kind: str                        # e.g. "tachycardia", "signal_loss"
    severity: str                    # "info" | "warning" | "critical"
    detail: str
    ts: float = field(default_factory=time.time)


@dataclass
class ProcessedBatch:
    """Aggregate output of the three Flink jobs + poisoning guard."""
    alarms: list[Alarm]
    coreset: list[CoresetPoint]
    dq_metrics: dict[str, float]     # data-quality summary
    rejected: int                    # points dropped by the poisoning guard
    dq_filtered: int = 0             # points dropped for low quality (not an attack)
    capped: int = 0                  # points dropped by the per-patient flood cap


# --------------------------------------------------------------------------- #
# Layer 5 <-> everyone downstream
# --------------------------------------------------------------------------- #
@dataclass
class StorageReceipt:
    """Confirmation that a batch landed across the multi-layered stores."""
    raw_lake_uri: str                # WORM object-lock URI
    raw_sha256: str                  # immutability anchor (21 CFR Part 11)
    tsdb_points: int                 # rows written to the hot time-series store
    feature_keys: list[str]          # online feature-store keys
    coreset_size: int                # points parked for training


# --------------------------------------------------------------------------- #
# Layer 7 output
# --------------------------------------------------------------------------- #
@dataclass
class ModelArtifact:
    name: str
    version: str
    weights: np.ndarray
    metrics: dict[str, float]
    dp_epsilon_spent: float          # privacy cost of DP-SGD training
    sbom: list[str]                  # software bill of materials
    stage: str                       # "shadow" | "production" | "rejected"
    signature: str                   # HMAC over the artifact


# --------------------------------------------------------------------------- #
# Layer 8 output
# --------------------------------------------------------------------------- #
@dataclass
class ComplianceReport:
    audit_chain_head: str            # head hash of the tamper-evident audit log
    audit_events: int
    privacy_budget_remaining: float
    soup_components: int             # IEC 62304 SOUP inventory size
    controls: dict[str, bool]        # named control -> pass/fail
    generated_ts: float = field(default_factory=time.time)
