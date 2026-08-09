"""
Layer 5 - STORAGE (multi-layered)
===============================

Five stores behind one facade:

  * ObjectLake        WORM raw archive, SHA-256 anchored (21 CFR Part 11)
  * TSDB              hot time-series store (recent window only)
  * FeatureStore      online features for serving/training
  * MetadataStore     patient/device/run registry
  * DPReleaseStore    differentially-private query boundary (Gaussian mechanism)

The design rule that keeps cost sane: the raw firehose lands ONLY in the cheap
WORM lake; the TSDB holds a small hot window; only the coreset feeds training.

Production swap-in: S3+Object Lock+Delta, Timestream-for-InfluxDB / Timescale,
Redis/Feast, Aurora Postgres, OpenDP/SmartNoise for the DP boundary.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from ..core.contracts import ProcessedBatch, StorageReceipt, StreamRecord
from ..crosscutting.privacy_budget import PrivacyBudgetLedger


class ObjectLake:
    """Append-only, hash-anchored WORM store (object-lock semantics)."""
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, key: str, blob: bytes) -> tuple[str, str]:
        if key in self._objects:                       # WORM: no overwrite
            raise PermissionError(f"object-lock: {key} is immutable")
        self._objects[key] = blob
        digest = hashlib.sha256(blob).hexdigest()
        return f"s3://raw-lake/{key}", digest


class TSDB:
    """Hot time-series store: bounded window of recent points."""
    def __init__(self, hot_window: int = 10_000) -> None:
        self._points: deque = deque(maxlen=hot_window)

    def write(self, patient_id: str, hr: float, sqi: float, ts: float) -> None:
        self._points.append((patient_id, hr, sqi, ts))

    def __len__(self) -> int:
        return len(self._points)


class FeatureStore:
    def __init__(self) -> None:
        self._features: dict[str, np.ndarray] = {}

    def upsert(self, key: str, vec: np.ndarray) -> None:
        self._features[key] = vec

    def keys(self) -> list[str]:
        return list(self._features)

    def get(self, key: str) -> np.ndarray:
        return self._features[key]


class MetadataStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def register_run(self, run_id: str, **attrs) -> None:
        self.rows[run_id] = attrs


class DPReleaseStore:
    """
    Differentially-private release boundary. Analytics/exports go through
    here; every query is noised (Gaussian mechanism) and debits the budget.
    """
    def __init__(self, ledger: PrivacyBudgetLedger):
        self.ledger = ledger

    def private_mean(self, values: list[float], sensitivity: float,
                     epsilon: float, delta: float = 1e-6) -> float:
        self.ledger.debit("dp_release_mean", epsilon, delta)
        if not values:
            return 0.0
        true_mean = float(np.mean(values))
        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon
        return true_mean + float(np.random.default_rng().normal(0, sigma))


class StorageLayer:
    """Facade wiring the five stores together for one ProcessedBatch."""

    def __init__(self, ledger: PrivacyBudgetLedger):
        self.lake = ObjectLake()
        self.tsdb = TSDB()
        self.features = FeatureStore()
        self.metadata = MetadataStore()
        self.dp = DPReleaseStore(ledger)
        self._seq = 0

    def persist(self, run_id: str, records: list[StreamRecord],
                batch: ProcessedBatch) -> StorageReceipt:
        # 1) raw -> WORM lake (immutably)
        raw = b"".join(r.packet.compressed for r in records)
        uri, digest = self.lake.put(f"{run_id}/{self._seq}.bin", raw)
        self._seq += 1

        # 2) hot -> TSDB (derived scalar series only)
        for r in records:
            n_beats = len(r.packet.qrs_indices)
            win_s = r.packet.n_samples / r.packet.fs_hz if r.packet.fs_hz else 1.0
            hr = 60.0 * n_beats / win_s if win_s else 0.0
            self.tsdb.write(r.packet.patient_id, hr, r.packet.sqi, r.ingest_ts)

        # 3) coreset -> feature store (parked for Layer 7 training)
        fkeys = []
        for i, cp in enumerate(batch.coreset):
            key = f"{run_id}:coreset:{i}"
            self.features.upsert(key, cp.embedding)
            fkeys.append(key)

        # 4) run metadata
        self.metadata.register_run(run_id, records=len(records),
                                   coreset=len(batch.coreset),
                                   alarms=len(batch.alarms))

        return StorageReceipt(raw_lake_uri=uri, raw_sha256=digest,
                              tsdb_points=len(self.tsdb),
                              feature_keys=fkeys,
                              coreset_size=len(batch.coreset))
